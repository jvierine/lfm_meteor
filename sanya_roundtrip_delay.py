import jcoord


SPEED_OF_LIGHT = 299792458.0
SANYA_LAT_DEG = 18.3492
SANYA_LON_DEG = 109.6222
SANYA_ALT_M = 50.0
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0
SANYA_RANGE_KM = 69.9
DANZHOU_LAT_DEG = 19.5281
DANZHOU_LON_DEG = 109.1322
DANZHOU_ALT_M = 99.9
WENCHANG_LAT_DEG = 19.5982
WENCHANG_LON_DEG = 110.7908
WENCHANG_ALT_M = 24.9


target_llh = jcoord.az_el_r2geodetic(
    SANYA_LAT_DEG,
    SANYA_LON_DEG,
    SANYA_ALT_M,
    SANYA_AZ_DEG,
    SANYA_EL_DEG,
    SANYA_RANGE_KM * 1e3,
)

sanya_ecef = jcoord.geodetic2ecef(SANYA_LAT_DEG, SANYA_LON_DEG, SANYA_ALT_M)
target_ecef = jcoord.geodetic2ecef(target_llh[0], target_llh[1], target_llh[2])
danzhou_ecef = jcoord.geodetic2ecef(DANZHOU_LAT_DEG, DANZHOU_LON_DEG, DANZHOU_ALT_M)
wenchang_ecef = jcoord.geodetic2ecef(WENCHANG_LAT_DEG, WENCHANG_LON_DEG, WENCHANG_ALT_M)

dx = target_ecef[0] - sanya_ecef[0]
dy = target_ecef[1] - sanya_ecef[1]
dz = target_ecef[2] - sanya_ecef[2]
one_way_m = (dx**2.0 + dy**2.0 + dz**2.0) ** 0.5

dx = target_ecef[0] - wenchang_ecef[0]
dy = target_ecef[1] - wenchang_ecef[1]
dz = target_ecef[2] - wenchang_ecef[2]
target_to_wenchang_m = (dx**2.0 + dy**2.0 + dz**2.0) ** 0.5

dx = target_ecef[0] - danzhou_ecef[0]
dy = target_ecef[1] - danzhou_ecef[1]
dz = target_ecef[2] - danzhou_ecef[2]
target_to_danzhou_m = (dx**2.0 + dy**2.0 + dz**2.0) ** 0.5

sanya_one_way_delay_us = one_way_m / SPEED_OF_LIGHT * 1e6
sanya_round_trip_delay_us = 2.0 * one_way_m / SPEED_OF_LIGHT * 1e6
danzhou_one_way_delay_us = target_to_danzhou_m / SPEED_OF_LIGHT * 1e6
wenchang_one_way_delay_us = target_to_wenchang_m / SPEED_OF_LIGHT * 1e6
sanya_to_danzhou_total_delay_us = (one_way_m + target_to_danzhou_m) / SPEED_OF_LIGHT * 1e6
sanya_to_wenchang_total_delay_us = (one_way_m + target_to_wenchang_m) / SPEED_OF_LIGHT * 1e6

print(f"One-way range: {one_way_m/1e3:.3f} km")
print(f"Sanya -> target delay: {sanya_one_way_delay_us:.2f} us")
print(f"Sanya -> target -> Sanya delay: {sanya_round_trip_delay_us:.2f} us")
print(f"Target to Danzhou: {target_to_danzhou_m/1e3:.3f} km")
print(f"Target -> Danzhou delay: {danzhou_one_way_delay_us:.2f} us")
print(f"Sanya -> target -> Danzhou delay: {sanya_to_danzhou_total_delay_us:.2f} us")
print(f"Target to Wenchang: {target_to_wenchang_m/1e3:.3f} km")
print(f"Target -> Wenchang delay: {wenchang_one_way_delay_us:.2f} us")
print(f"Sanya -> target -> Wenchang delay: {sanya_to_wenchang_total_delay_us:.2f} us")
