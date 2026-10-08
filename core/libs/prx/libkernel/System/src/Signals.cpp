#include "prx/libc/include/General.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libkernel/Pthread/include/Pthread.hpp"
#include <array>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <mutex>
#include <stdexcept>

extern "C" int* APS5_VABI __error_nid_postfix();
extern "C" Pthread APS5_VABI scePthreadSelf();
#ifdef _WIN32
extern "C" int APS5_VABI sceKernelRaiseException(Pthread thread, int signum);
#endif
namespace {
using GuestHandler = void (APS5_VABI *)(int);
std::atomic<GuestHandler> handlers[32]{};
static_assert(std::atomic<GuestHandler>::is_always_lock_free);
std::mutex registration;
constexpr std::array<int, 6> MappedSignals{2, 4, 6, 8, 11, 15};
constexpr std::array<int, 7> MaskedSignals{2, 4, 6, 8, 11, 15, GUEST_RAISED_SIGNAL};
constexpr std::uint32_t UnblockableSignals = GuestSignalBit(9) | GuestSignalBit(17) | GuestSignalBit(32);
using GuestMask = std::array<std::uint32_t, 4>;
int NativeSignal(int guest) {
    switch (guest) {
        case 2: return SIGINT;
        case 4: return SIGILL;
        case 6: return SIGABRT;
        case 8: return SIGFPE;
        case 11: return SIGSEGV;
        case 15: return SIGTERM;
        default: return 0;
    }
}
void Dispatch(int native) {
    int guest = 0;
    for (int candidate : MappedSignals)
        if (NativeSignal(candidate) == native) { guest = candidate; break; }
    if (!guest) return;
#ifdef _WIN32
    // Preserve the guest's persistent registration across CRT delivery.
    std::signal(native, Dispatch);
    auto* self = scePthreadSelf();
    if (GuestSignalBlocked(*self, guest)) {
        self->pendingSignals.fetch_or(GuestSignalBit(guest));
        return;
    }
#endif
    const auto callback = handlers[guest].load();
    if (reinterpret_cast<std::uintptr_t>(callback) > 1) callback(guest);
}

#ifdef _WIN32
GuestMask CurrentMask(const PthreadPrivate& self) {
    GuestMask mask{};
    for (std::size_t word = 0; word < mask.size(); ++word) mask[word] = self.signalMask[word].load();
    return mask;
}

void ApplyMask(PthreadPrivate& self, const GuestMask& mask) {
    for (const int guest : MaskedSignals) {
        const std::uint32_t bit = GuestSignalBit(guest);
        if ((mask[0] & bit) != 0 || (self.pendingSignals.fetch_and(~bit) & bit) == 0) continue;
        if (guest == GUEST_RAISED_SIGNAL) sceKernelRaiseException(&self, guest);
        else std::raise(NativeSignal(guest));
    }
}
#else
int HostSignal(int guest) {
    return guest == GUEST_RAISED_SIGNAL ? HOST_RAISED_SIGNAL : NativeSignal(guest);
}

GuestMask CurrentMask(const PthreadPrivate& self) {
    GuestMask mask{};
    for (std::size_t word = 0; word < mask.size(); ++word) mask[word] = self.signalMask[word].load();
    sigset_t host;
    pthread_sigmask(SIG_BLOCK, nullptr, &host);
    for (const int guest : MaskedSignals) {
        if (sigismember(&host, HostSignal(guest)) == 1) mask[0] |= GuestSignalBit(guest);
        else mask[0] &= ~GuestSignalBit(guest);
    }
    return mask;
}

void ApplyMask(PthreadPrivate&, const GuestMask& mask) {
    sigset_t blocked;
    sigset_t unblocked;
    sigemptyset(&blocked);
    sigemptyset(&unblocked);
    for (const int guest : MaskedSignals)
        sigaddset((mask[0] & GuestSignalBit(guest)) != 0 ? &blocked : &unblocked, HostSignal(guest));
    pthread_sigmask(SIG_BLOCK, &blocked, nullptr);
    pthread_sigmask(SIG_UNBLOCK, &unblocked, nullptr);
}
#endif
}

struct GuestSignalSet {
    std::uint32_t bits[4];
};

extern "C" {
GuestHandler APS5_VABI signal_nid_postfix(int guest, GuestHandler handler) {
    const auto invalid = reinterpret_cast<GuestHandler>(static_cast<std::uintptr_t>(-1));
    const int native = NativeSignal(guest);
    if (!native || handler == invalid) { *__error_nid_postfix() = 22; return invalid; }
    std::lock_guard lock(registration);
    const auto previous = handlers[guest].exchange(handler);
    const auto address = reinterpret_cast<std::uintptr_t>(handler);
    auto hostHandler = address == 0 ? SIG_DFL : address == 1 ? SIG_IGN : Dispatch;
    if (std::signal(native, hostHandler) == SIG_ERR) {
        handlers[guest].store(previous);
        *__error_nid_postfix() = 22;
        return invalid;
    }
    return previous;
}
int APS5_VABI raise_nid_postfix(int guest) {
    const int native = NativeSignal(guest);
    if (!native) { *__error_nid_postfix() = 22; return -1; }
#ifdef _WIN32
    auto* self = scePthreadSelf();
    if (GuestSignalBlocked(*self, guest)) {
        self->pendingSignals.fetch_or(GuestSignalBit(guest));
        return 0;
    }
#endif
    const int result = std::raise(native);
    if (result) *__error_nid_postfix() = 22;
    return result ? -1 : 0;
}
int APS5_VABI _sigprocmask_nid_postfix(int how, const GuestSignalSet* set, GuestSignalSet* previousSet) {
    if (set != nullptr && (how < 1 || how > 3)) throw std::invalid_argument("_sigprocmask: invalid how");
    auto* self = scePthreadSelf();
    const GuestMask previous = CurrentMask(*self);
    GuestMask next = previous;
    if (set != nullptr) {
        for (std::size_t word = 0; word < next.size(); ++word) {
            switch (how) {
                case 1: next[word] |= set->bits[word]; break;
                case 2: next[word] &= ~set->bits[word]; break;
                default: next[word] = set->bits[word]; break;
            }
        }
        next[0] &= ~UnblockableSignals;
        for (std::size_t word = 0; word < next.size(); ++word) self->signalMask[word].store(next[word]);
    }
    if (previousSet != nullptr)
        for (std::size_t word = 0; word < previous.size(); ++word) previousSet->bits[word] = previous[word];
    if (set != nullptr) ApplyMask(*self, next);
    return 0;
}

int APS5_VABI sigprocmask_nid_postfix(int how, const void* set, void* previousSet) {
    return _sigprocmask_nid_postfix(how, static_cast<const GuestSignalSet*>(set),
                                    static_cast<GuestSignalSet*>(previousSet));
}
}

extern "C" {

int APS5_VABI _is_signal_return_nid_postfix(std::uint64_t programCounter) {
    (void)programCounter;
    return 0;
}

}
