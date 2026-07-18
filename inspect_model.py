import os
import joblib
print('cwd', os.getcwd())
print('model dir', os.listdir('model'))
try:
    obj = joblib.load('model/fraud_pipeline.pkl')
    print(type(obj))
    print(obj)
except Exception as e:
    print('LOAD_ERROR', repr(e))
