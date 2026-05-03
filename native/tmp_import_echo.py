import sys,traceback
sys.path.insert(0,'native')
try:
    import echo_host_V2 as h
    print('Imported OK')
except Exception as e:
    traceback.print_exc()
    print('Error:',e)
