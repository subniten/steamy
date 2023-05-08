clear
clc
close all

file = "230504_STA_";
Uload = load(file + "east.mat");
Vload = load(file + "north.mat");
tarray = datetime(2000+Vload.ConYear, Vload.ConMonth, Vload.ConDay, Vload.ConHour, Vload.ConMin, Vload.ConSec);
U = Uload.ConVel;
V = Vload.ConVel;
U(U< -1000) = NaN;  V(V< -1000) = NaN;  

bins = Uload.ConBins;   binSize = Uload.RDIBinSize; Z = (bins-1) * binSize + Uload.RDIBin1Mid;

Lat = load(file + "ancillary.mat").AnLLatDeg;
Lon = load(file + "ancillary.mat").AnLLonDeg;

Umean = mean(U, 2, 'omitnan');
Vmean = mean(V, 2, 'omitnan');

% Bathymetry File
ncid = netcdf.open("gebco_2022_n60.0_s54.0_w7.5_e15.0.nc");
ncdisp("gebco_2022_n60.0_s54.0_w7.5_e15.0.nc")
bat_lat = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'lat'));
bat_lon = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'lon'));
bat_elev = netcdf.getVar(ncid, netcdf.inqVarID(ncid,'elevation'));

% Surface currents
figure(1)
hold on
grid on
title('Surface Currents')
xlim([10.2 11.6])
ylim([57.6 58.1])
quiver(Lon, Lat, U(:,1), V(:,1), 5)
contourf(bat_lon, bat_lat, bat_elev', [0 1], 'k');
colormap("bone")

% Depth Averaged currents
figure(2)
hold on
grid on
title('Depth Averaged Currents')
xlim([10.2 11.6])
ylim([57.6 58.1])
quiver(Lon, Lat, Umean, Vmean, 5)
contourf(bat_lon, bat_lat, bat_elev', [0 1], 'k');
colormap("bone")

% figure(1)
% hold on
% ylim([0 100])
% p = pcolor(tarray, Z, V');
% set(p, 'EdgeColor', 'none')
% set(gca, 'yDir', 'reverse')
% colorbar