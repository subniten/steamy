import pandas
import xarray

_renamer = dict(
    lat='latitude',
    long='longitude',
    ts='time',
    airtemp='temperature',
    airpressure='pressure',
)


def read_met_file(met_file_path):
    df = (
        pandas.read_csv(met_file_path, sep=',', parse_dates=['ts'])
        .rename(columns=_renamer)
        .rename_axis(index=['sample'])
        .to_xarray()
    )
    return df.assign_coords(
        dict(latitude=df.latitude, longitude=df.longitude, time=df.time)
    )


def read_met_directory(met_directory_path):
    _ = xarray.concat(
        [read_met_file(fp) for fp in sorted(met_directory_path.glob('*.csv'))],
        dim='sample',
    )
