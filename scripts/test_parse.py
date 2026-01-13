from utils.validation import parse_json_strict

samples = [
    'prefix {"tool":"add","args":{"a":1,"b":2}} suffix',
    '```json\n{"tool":"multiply","args":{"a":3,"b":4}}\n```',
    '{"not_an_object": [1,2,3]}',
]

for s in samples:
    try:
        print('INPUT:', s)
        print('PARSED:', parse_json_strict(s))
    except Exception as e:
        print('ERROR:', e)
    print('---')
