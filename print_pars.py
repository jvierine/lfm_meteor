SPEED_OF_LIGHT = 299792458.0


# Original first-sample delay calculations.
# Sanya uses r0 = 69.9 km monostatic.
# Danzhou and Wenchang use the equivalent-path offsets from the original analysis.

sanya_r0_km = 69.9
danzhou_equiv_r0_km = 69.9 + 28.1378
wenchang_equiv_r0_km = 69.9 + 45.7489


def delay_us(equivalent_range_km):
    return 2.0 * equivalent_range_km * 1e3 / SPEED_OF_LIGHT * 1e6


print("Sanya time of first sample %1.2f (microseconds)" % (delay_us(sanya_r0_km)))
print("Danzhou time of first sample %1.2f (microseconds)" % (delay_us(danzhou_equiv_r0_km)))
print("Wenchang time of first sample %1.2f (microseconds)" % (delay_us(wenchang_equiv_r0_km)))
