function [alt_cross,rg_t,rg_r,rg_delay]=rangedelay(lat_t,lon_t,alt_t,az_t,el_t,lat_r,lon_r,alt_r,az_r,el_r)
% alt_cross 共体大地高度（km）
% rg_t      共体到发射站的距离（km）
% rg_r      共体到接收站的距离（km）
% rg_delay  接收站真实距离延迟（km）
% lat_t     发射站大地纬度    （°）
% lon_t     发射站大地经度    （°）
% alt_t     发射站大地高度    （km）
% az_t      发射站方位角      （°）
% el_t      发射站仰角        （°）
% lat_r     接收站大地纬度    （°）
% lon_r     接收站大地经度    （°）
% alt_r     接收站大地高度    （km）
% az_r      接收站方位角      （°）
% el_r      接收站仰角        （°）

% lat_t=18.3492;
% lon_t=109.6222;
% alt_t=0.05;
% az_t =10;
% el_t =75;
% lat_r=19.5281;
% lon_r=109.7908;
% alt_r=0.0249;
% az_r =226.7;
% el_r =30.56;

wgs=wgs84Ellipsoid('km');
[lat0,lon0,alt0]=aer2geodetic(az_t,el_t,50:5:2000,lat_t,lon_t,alt_t,wgs);
[a,e,r]=geodetic2aer(lat0,lon0,alt0,lat_r,lon_r,alt_r,wgs);
if all(diff(e)>0) || all(diff(e)<0)
    alt_cross=interp1(e,alt0,el_r);
elseif all(diff(a)>0) || all(diff(a)<0)
    alt_cross=interp1(a,alt0,az_r);
end
% ind=find(e==max(e));
% alt_cross=interp1(e(1:ind),alt0(1:ind),el_r);
% if alt_cross<50 || alt_cross>1250
%     % alt_cross=interp1(a,alt0,az_r);
% end
lat_cross=interp1(alt0,lat0,alt_cross);
lon_cross=interp1(alt0,lon0,alt_cross);
[~,~,rg_t]=geodetic2aer(lat_cross,lon_cross,alt_cross,lat_t,lon_t,alt_t,wgs);
[~,~,rg_r]=geodetic2aer(lat_cross,lon_cross,alt_cross,lat_r,lon_r,alt_r,wgs);
rg_delay=rg_r-(rg_r+rg_t)/2;
end


