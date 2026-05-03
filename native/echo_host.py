#!/usr/bin/env python3
import sys
import json
import struct
import time

# Native Messaging helper functions (Chrome/Firefox protocol)

def read_message():
    raw_length = sys.stdin.buffer.read(4)
    if len(raw_length) == 0:
        return None
    message_length = struct.unpack('<I', raw_length)[0]
    message = sys.stdin.buffer.read(message_length).decode('utf-8')
    return json.loads(message)


def send_message(message):
    encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('<I', len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def classify_items(envelope):
    req_id = envelope.get('requestId')
    payload = envelope.get('payload') or {}
    items = payload.get('items') or []
    model = payload.get('model') or {}

    # Dummy logic: return neutral scores for every item
    results = []
    for it in items:
        modality = it.get('modality')
        results.append({
            'id': it.get('id'),
            'modality': modality,
            'label': 'uncertain',
            'score': 0.5,
            'model': model.get(modality, 'echo-dev'),
            'durationMs': 1,
            'notes': None
        })
    return {
        'version': 1,
        'type': 'classifyResult',
        'requestId': req_id,
        'timestamp': int(time.time() * 1000),
        'results': results,
        'errors': []
    }


def main():
    while True:
        msg = read_message()
        if msg is None:
            break
        mtype = msg.get('type')
        if mtype == 'ping' or mtype == 'PING':
            send_message({'ok': True, 'time': int(time.time() * 1000)})
        elif mtype == 'classify':
            resp = classify_items(msg)
            send_message(resp)
        else:
            send_message({'ok': False, 'error': f'Unknown type: {mtype}'})


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Best-effort error signaling; real hosts should log to a file
        try:
            send_message({'ok': False, 'error': str(e)})
        except Exception:
            pass
