"""Reading the shared memory of Assetto Corsa Competizione.

The layout follows the ACC Shared Memory Documentation, version 1.8.12. The simulator
publishes three pages: physics (every physics step), graphics (session and lap state) and
static (car and track). Text fields are UTF-16 and are declared as arrays of 16-bit values
so that the sizes are the same on every platform.

Only fields that the study needs are copied into a sample. The name of the player, which is
part of the static page, is never read.
"""
from __future__ import annotations

import ctypes
import mmap
import sys
import time
from ctypes import c_float, c_int, c_uint16

STATUS_OFF, STATUS_REPLAY, STATUS_LIVE, STATUS_PAUSE = 0, 1, 2, 3


class Physics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int), ("gas", c_float), ("brake", c_float), ("fuel", c_float), ("gear", c_int),
        ("rpms", c_int), ("steerAngle", c_float), ("speedKmh", c_float), ("velocity", c_float * 3),
        ("accG", c_float * 3), ("wheelSlip", c_float * 4), ("wheelLoad", c_float * 4),
        ("wheelsPressure", c_float * 4), ("wheelAngularSpeed", c_float * 4), ("tyreWear", c_float * 4),
        ("tyreDirtyLevel", c_float * 4), ("tyreCoreTemperature", c_float * 4), ("camberRAD", c_float * 4),
        ("suspensionTravel", c_float * 4), ("drs", c_float), ("tc", c_float), ("heading", c_float),
        ("pitch", c_float), ("roll", c_float), ("cgHeight", c_float), ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int), ("pitLimiterOn", c_int), ("abs", c_float), ("kersCharge", c_float),
        ("kersInput", c_float), ("autoShifterOn", c_int), ("rideHeight", c_float * 2), ("turboBoost", c_float),
        ("ballast", c_float), ("airDensity", c_float), ("airTemp", c_float), ("roadTemp", c_float),
        ("localAngularVel", c_float * 3), ("finalFF", c_float), ("performanceMeter", c_float),
        ("engineBrake", c_int), ("ersRecoveryLevel", c_int), ("ersPowerLevel", c_int), ("ersHeatCharging", c_int),
        ("ersIsCharging", c_int), ("kersCurrentKJ", c_float), ("drsAvailable", c_int), ("drsEnabled", c_int),
        ("brakeTemp", c_float * 4), ("clutch", c_float), ("tyreTempI", c_float * 4), ("tyreTempM", c_float * 4),
        ("tyreTempO", c_float * 4), ("isAIControlled", c_int), ("tyreContactPoint", (c_float * 3) * 4),
        ("tyreContactNormal", (c_float * 3) * 4), ("tyreContactHeading", (c_float * 3) * 4),
        ("brakeBias", c_float), ("localVelocity", c_float * 3), ("P2PActivations", c_int), ("P2PStatus", c_int),
        ("currentMaxRpm", c_int), ("mz", c_float * 4), ("fx", c_float * 4), ("fy", c_float * 4),
        ("slipRatio", c_float * 4), ("slipAngle", c_float * 4), ("tcInAction", c_int), ("absInAction", c_int),
        ("suspensionDamage", c_float * 4), ("tyreTemp", c_float * 4), ("waterTemp", c_float),
        ("brakePressure", c_float * 4), ("frontBrakeCompound", c_int), ("rearBrakeCompound", c_int),
        ("padLife", c_float * 4), ("discLife", c_float * 4), ("ignitionOn", c_int), ("starterEngineOn", c_int),
        ("isEngineRunning", c_int), ("kerbVibration", c_float), ("slipVibrations", c_float),
        ("gVibrations", c_float), ("absVibrations", c_float),
    ]


class Graphics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int), ("status", c_int), ("session", c_int), ("currentTime", c_uint16 * 15),
        ("lastTime", c_uint16 * 15), ("bestTime", c_uint16 * 15), ("split", c_uint16 * 15),
        ("completedLaps", c_int), ("position", c_int), ("iCurrentTime", c_int), ("iLastTime", c_int),
        ("iBestTime", c_int), ("sessionTimeLeft", c_float), ("distanceTraveled", c_float), ("isInPit", c_int),
        ("currentSectorIndex", c_int), ("lastSectorTime", c_int), ("numberOfLaps", c_int),
        ("tyreCompound", c_uint16 * 33), ("replayTimeMultiplier", c_float), ("normalizedCarPosition", c_float),
        ("activeCars", c_int), ("carCoordinates", (c_float * 3) * 60), ("carID", c_int * 60),
        ("playerCarID", c_int), ("penaltyTime", c_float), ("flag", c_int), ("penalty", c_int),
        ("idealLineOn", c_int), ("isInPitLane", c_int), ("surfaceGrip", c_float), ("mandatoryPitDone", c_int),
        ("windSpeed", c_float), ("windDirection", c_float), ("isSetupMenuVisible", c_int),
        ("mainDisplayIndex", c_int), ("secondaryDisplayIndex", c_int), ("TC", c_int), ("TCCut", c_int),
        ("EngineMap", c_int), ("ABS", c_int), ("fuelXLap", c_float), ("rainLights", c_int),
        ("flashingLights", c_int), ("lightsStage", c_int), ("exhaustTemperature", c_float), ("wiperLV", c_int),
        ("DriverStintTotalTimeLeft", c_int), ("DriverStintTimeLeft", c_int), ("rainTyres", c_int),
        ("sessionIndex", c_int), ("usedFuel", c_float), ("deltaLapTime", c_uint16 * 15), ("iDeltaLapTime", c_int),
        ("estimatedLapTime", c_uint16 * 15), ("iEstimatedLapTime", c_int), ("isDeltaPositive", c_int),
        ("iSplit", c_int), ("isValidLap", c_int), ("fuelEstimatedLaps", c_float), ("trackStatus", c_uint16 * 33),
        ("missingMandatoryPits", c_int), ("Clock", c_float), ("directionLightsLeft", c_int),
        ("directionLightsRight", c_int), ("GlobalYellow", c_int), ("GlobalYellow1", c_int),
        ("GlobalYellow2", c_int), ("GlobalYellow3", c_int), ("GlobalWhite", c_int), ("GlobalGreen", c_int),
        ("GlobalChequered", c_int), ("GlobalRed", c_int), ("mfdTyreSet", c_int), ("mfdFuelToAdd", c_float),
        ("mfdTyrePressureLF", c_float), ("mfdTyrePressureRF", c_float), ("mfdTyrePressureLR", c_float),
        ("mfdTyrePressureRR", c_float), ("trackGripStatus", c_int), ("rainIntensity", c_int),
        ("rainIntensityIn10min", c_int), ("rainIntensityIn30min", c_int), ("currentTyreSet", c_int),
        ("strategyTyreSet", c_int),
    ]


class Static(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("smVersion", c_uint16 * 15), ("acVersion", c_uint16 * 15), ("numberOfSessions", c_int),
        ("numCars", c_int), ("carModel", c_uint16 * 33), ("track", c_uint16 * 33), ("playerName", c_uint16 * 33),
        ("playerSurname", c_uint16 * 33), ("playerNick", c_uint16 * 33), ("sectorCount", c_int),
        ("maxTorque", c_float), ("maxPower", c_float), ("maxRpm", c_int), ("maxFuel", c_float),
        ("suspensionMaxTravel", c_float * 4), ("tyreRadius", c_float * 4), ("maxTurboBoost", c_float),
        ("deprecated_1", c_float), ("deprecated_2", c_float), ("penaltiesEnabled", c_int),
        ("aidFuelRate", c_float), ("aidTireRate", c_float), ("aidMechanicalDamage", c_float),
        ("aidAllowTyreBlankets", c_int), ("aidStability", c_float), ("aidAutoClutch", c_int),
        ("aidAutoBlip", c_int), ("hasDRS", c_int), ("hasERS", c_int), ("hasKERS", c_int), ("kersMaxJ", c_float),
        ("engineBrakeSettingsCount", c_int), ("ersPowerControllerCount", c_int), ("trackSPlineLength", c_float),
        ("trackConfiguration", c_uint16 * 33), ("ersMaxJ", c_float), ("isTimedRace", c_int),
        ("hasExtraLap", c_int), ("carSkin", c_uint16 * 33), ("reversedGridPositions", c_int),
        ("PitWindowStart", c_int), ("PitWindowEnd", c_int), ("isOnline", c_int), ("dryTyresName", c_uint16 * 33),
        ("wetTyresName", c_uint16 * 33),
    ]


def text(field) -> str:
    """Decode a UTF-16 text field up to its terminating zero."""
    chars = []
    for value in field:
        if value == 0:
            break
        chars.append(chr(value))
    return "".join(chars)


def sample_from(physics: Physics, graphics: Graphics, t: float) -> dict:
    """The values of one moment that go into the session file."""
    car = 0
    for k in range(min(max(graphics.activeCars, 1), 60)):
        if graphics.carID[k] == graphics.playerCarID:
            car = k
            break
    x, y, z = graphics.carCoordinates[car]
    ws, sr, bp = physics.wheelSlip, physics.slipRatio, physics.brakePressure
    return {
        "t": t, "packet_id": physics.packetId, "lap": graphics.completedLaps,
        "norm_pos": graphics.normalizedCarPosition, "speed_kmh": physics.speedKmh, "gas": physics.gas,
        "brake": physics.brake, "steer": physics.steerAngle, "gear": physics.gear, "rpm": physics.rpms,
        "acc_x": physics.accG[0], "acc_y": physics.accG[1], "acc_z": physics.accG[2],
        "wslip_fl": ws[0], "wslip_fr": ws[1], "wslip_rl": ws[2], "wslip_rr": ws[3],
        "sratio_fl": sr[0], "sratio_fr": sr[1], "sratio_rl": sr[2], "sratio_rr": sr[3],
        "bpress_fl": bp[0], "bpress_fr": bp[1], "bpress_rl": bp[2], "bpress_rr": bp[3],
        "yaw_rate": physics.localAngularVel[1], "tc_on": float(physics.tcInAction),
        "abs_on": float(physics.absInAction), "tyres_out": physics.numberOfTyresOut,
        "in_pit": graphics.isInPit, "in_pit_lane": graphics.isInPitLane, "valid_lap": graphics.isValidLap,
        "cur_time_ms": graphics.iCurrentTime, "last_time_ms": graphics.iLastTime, "fuel": physics.fuel,
        "x": x, "y": y, "z": z,
    }


class AccSource:
    """Live connection to the simulator. Works on Windows only."""

    name = "acc"

    # Bytes mapped per page. The static page is mapped a little shorter than the documented
    # structure, because older game versions publish a shorter page and mapping more than
    # exists fails. The fields that are read all lie inside this range.
    SIZES = {"physics": ctypes.sizeof(Physics), "graphics": ctypes.sizeof(Graphics), "static": 784}

    def __init__(self):
        if sys.platform != "win32":
            raise OSError("The shared memory of ACC is only available on Windows.")
        self._maps = {key: mmap.mmap(-1, size, f"Local\\acpmf_{key}", access=mmap.ACCESS_READ)
                      for key, size in self.SIZES.items()}

    def _read(self, key, struct):
        buffer = self._maps[key]
        buffer.seek(0)
        raw = buffer.read(self.SIZES[key])
        return struct.from_buffer_copy(raw.ljust(ctypes.sizeof(struct), b"\0"))

    def status(self) -> int:
        return self._read("graphics", Graphics).status

    def info(self) -> dict:
        """Car, track and assist settings. Personal fields of the page are left out."""
        s = self._read("static", Static)
        return {"car_model": text(s.carModel), "track": text(s.track), "acc_version": text(s.acVersion),
                "sm_version": text(s.smVersion), "max_rpm": s.maxRpm, "max_fuel": s.maxFuel,
                "aid_stability": s.aidStability, "aid_auto_clutch": s.aidAutoClutch, "aid_auto_blip": s.aidAutoBlip,
                "aid_fuel_rate": s.aidFuelRate, "aid_tyre_rate": s.aidTireRate,
                "aid_mechanical_damage": s.aidMechanicalDamage}

    def settings(self) -> dict:
        """Settings and conditions at the start of a recording. They have to be equal across sessions."""
        g = self._read("graphics", Graphics)
        p = self._read("physics", Physics)
        return {"tc_level": g.TC, "abs_level": g.ABS, "engine_map": g.EngineMap, "ideal_line_on": g.idealLineOn,
                "tyre_compound": text(g.tyreCompound), "session_type": g.session, "rain_tyres": g.rainTyres,
                "track_grip_status": g.trackGripStatus, "rain_intensity": g.rainIntensity,
                "surface_grip": round(g.surfaceGrip, 3), "wind_speed": round(g.windSpeed, 1),
                "air_temp": round(p.airTemp, 1), "road_temp": round(p.roadTemp, 1),
                "brake_bias": round(p.brakeBias, 3), "fuel_at_start": round(p.fuel, 1)}

    def read(self) -> tuple[int, dict]:
        graphics = self._read("graphics", Graphics)
        physics = self._read("physics", Physics)
        return graphics.status, sample_from(physics, graphics, time.perf_counter())

    def close(self) -> None:
        for buffer in self._maps.values():
            buffer.close()
