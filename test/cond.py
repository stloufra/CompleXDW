from mpmath import mp, mpf, fabs
import random

def generate_abcd_mp(K, precision_digits=60, max_tries=10000):
    mp.dps = precision_digits
    K = mpf(K)
    N = random.uniform(1,10000)

    invalid_tries = 0

    for attempt in range(max_tries):
        a = mpf(random.uniform(-N, N))
        b = mpf(random.uniform(-N, N))
        c = mpf(random.uniform(-N, N))

        if b == 0 or a == 0 or c == 0:
            invalid_tries += 1
            continue

        q = a * c
        i = random.randint(0, 3)

        if i == 0:
            p = (fabs(q) - K * q) / (K - 1)
        elif i == 1:
            p = (-fabs(q) - K * q) / (K + 1)
        elif i == 2:
            p = (fabs(q) - K * q) / (K + 1)
        elif i == 3:
            p = (-fabs(q) - K * q) / (K - 1)

        valid = (
            (i == 0 and p >= 0 and q + p >= 0) or
            (i == 1 and p >= 0 and q + p <  0) or
            (i == 2 and p <  0 and q + p >= 0) or
            (i == 3 and p <  0 and q + p <  0)
        )

        if not valid:
            invalid_tries += 1
            continue

        d = p / b
        denom = fabs(q + p)
        if denom == 0:
            invalid_tries += 1
            continue

        K_check = (fabs(q) + fabs(p)) / denom
        if fabs(K_check - K) / K < mpf('1e-9'):
            return a, b, c, d, K_check, invalid_tries, attempt + 1

        invalid_tries += 1

    return None


header = (f"{'K':>10} {'a':>14} {'b':>14} {'c':>14} {'d':>14} "
          f"{'K_check':>15} {'total':>7} {'invalid':>8} {'match':>6}")
print(header)
print("-" * 105)

for e in range(1, 41):
    K = 10**e
    prec = e + 20

    result = generate_abcd_mp(K, precision_digits=prec)
    if result:
        a, b, c, d, K_check, invalid_tries, total_tries = result
        print(f"{'10^'+str(e):>10} {float(a):>14.7f} {float(b):>14.7f} "
              f"{float(c):>14.7f} {float(d):>14.7f} "
              f"{float(K_check):>15.6e} {total_tries:>7} {invalid_tries:>8}   OK")
    else:
        print(f"{'10^'+str(e):>10} {'---':>14} {'---':>14} {'---':>14} {'---':>14} "
              f"{'No solution':>15} {'10000':>7} {'N/A':>8}  FAIL")
