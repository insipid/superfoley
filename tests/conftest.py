import pytest

from tests.fixtures.make_fixture import make_fixture


@pytest.fixture(scope="session")
def fixture(tmp_path_factory):
    """Synthetic video + truth, generated once per test session (seed 0)."""
    return make_fixture(tmp_path_factory.mktemp("fixture"), seed=0)


@pytest.fixture(scope="session")
def small_clips(fixture, tmp_path_factory):
    """Tiny 16 s derivatives of the fixture: no-audio, VFR, AV1 (None if no encoder)."""
    from tests.helpers import ff, has_encoder

    d = tmp_path_factory.mktemp("clips")
    src = fixture.video
    clips = {}
    clips["noaudio"] = d / "noaudio.mp4"
    ff("-i", src, "-t", 16, "-an", "-c:v", "copy", clips["noaudio"])
    # drop irregular frames but keep the original timestamps -> variable frame rate
    clips["vfr"] = d / "vfr.mp4"
    ff("-i", src, "-t", 16, "-vf", "select='not(eq(mod(n,7),3)+eq(mod(n,11),5))'",
       "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy",
       clips["vfr"])
    clips["av1"] = None
    for enc, extra in (("libsvtav1", ["-preset", "12"]), ("libaom-av1", ["-cpu-used", "8"])):
        if has_encoder(enc):
            clips["av1"] = d / "av1.mp4"
            ff("-i", src, "-t", 16, "-c:v", enc, *extra, "-c:a", "copy", clips["av1"])
            break
    return clips
