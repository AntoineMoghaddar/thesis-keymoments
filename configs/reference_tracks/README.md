# Reference track widths

`Monza.csv` and `Budapest.csv` are copied unchanged from the racetrack database of the Institute of
Automotive Technology, Technical University of Munich:
https://github.com/TUMFTM/racetrack-database (commit e59595d, 18 September 2021), licensed under the
GNU Lesser General Public License v3.0 (see `LICENSE` in this folder).

Columns: centre line `x_m`, `y_m`, track width to the right `w_tr_right_m` and to the left `w_tr_left_m`.
According to the authors the centre lines come from OpenStreetMap and the widths from satellite images,
and the quality varies per location.

In this project the files are used for one purpose: `km track` prints their track width and length next to
the values measured from the survey laps, as an independent check. They are not used to draw the track.
