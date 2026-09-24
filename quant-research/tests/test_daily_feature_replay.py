import importlib.util
import zlib
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('daily_feature_pilot',
    Path(__file__).resolve().parents[1]/'scripts/daily_feature_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_compressed_source_frame_rejects_truncation_and_bad_trailer():
    data = zlib.compress('0\x01success\x01示例'.encode())
    header = f'00.9.00\x0196\x01{len(data):010}'.encode()
    raw = header+data+pilot.END
    assert pilot.decode(raw) == ('96', ['0', 'success', '示例'])
    for corrupted in [raw[:-1], header+data+b'bad'+pilot.END,
                      header+data[:-1]+pilot.END]:
        with pytest.raises((ValueError, zlib.error)):
            pilot.decode(corrupted)


def test_login_checksum_rejects_changed_payload():
    text = '0\x01success\x01login\x01anonymous'
    prefix = f'00.9.00\x0101\x01{len(text):010}{text}'.encode()
    raw = prefix+b'\x01'+str(zlib.crc32(prefix)).encode()+pilot.END
    assert pilot.decode(raw)[1][2:] == ['login', 'anonymous']
    with pytest.raises(ValueError, match='checksum'):
        pilot.decode(raw.replace(b'anonymous', b'Anonymous'))


def test_failed_recovery_never_exposes_a_capture(tmp_path):
    import json
    (tmp_path/'job.json').write_text(json.dumps({'status':'failed','selected_attempt':None}))
    with pytest.raises(ValueError, match='Incomplete recovery'):
        pilot.replay(tmp_path)


def test_recovery_rejects_changed_attempt_manifest(tmp_path):
    import json
    attempt = tmp_path/'attempt-01'
    attempt.mkdir()
    (attempt/'manifest.json').write_text('{}')
    (tmp_path/'job.json').write_text(json.dumps({'status':'completed','selected_attempt':'attempt-01',
        'attempts':[{'directory':'attempt-01','manifest_sha256':'wrong','status':'completed'}]}))
    with pytest.raises(ValueError, match='Changed recovery'):
        pilot.replay(tmp_path)
