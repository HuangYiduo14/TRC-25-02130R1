"""One logical E2 forecast owner; all initialized reads fail closed."""
import math
import os
import time
from .common import root, read_json, write_json, digest


def validate(c, value):
    if not isinstance(value, dict):
        raise ValueError('E2 forecast is not an object')
    body={k:v for k,v in value.items() if k!='payload_sha256'}
    if value.get('payload_sha256')!=digest(body):
        raise ValueError('E2 forecast payload hash mismatch')
    if value.get('run_id')!=c['run_id'] or value.get('episode_id')!=c['episode_id']:
        raise ValueError('E2 forecast identity mismatch')
    for key in ['remaining_seconds','updated_unix']:
        number=value.get(key)
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or number<0:
            raise ValueError('Invalid E2 forecast '+key)
    if type(value.get('generation')) is not int or value['generation']<1:
        raise ValueError('Invalid E2 forecast generation')
    if value['updated_unix']>time.time()+60:
        raise ValueError('E2 forecast timestamp is in the future')
    if value.get('hard_deadline_unix')!=1788970386.:
        raise ValueError('E2 forecast cannot reset deadline')
    return value


def read(c):
    return validate(c,read_json(root(c)/'control/finalization_forecast.json'))


def publish(c, remaining_seconds, basis, *, initialize=False):
    path=root(c)/'control/finalization_forecast.json'
    if initialize:
        if path.exists():raise FileExistsError('E2 forecast already initialized')
        generation=1
    else:
        generation=read(c)['generation']+1
    body=dict(run_id=c['run_id'],episode_id=c['episode_id'],generation=generation,
              updated_unix=time.time(),remaining_seconds=remaining_seconds,basis=basis,
              logical_owner='E2_CAMPAIGN_COORDINATOR',writer_pid=os.getpid(),hard_deadline_unix=c['hard_deadline_unix'])
    value=dict(**body,payload_sha256=digest(body))
    validate(c,value)
    write_json(path,value)
    if read(c)!=value:raise ValueError('E2 forecast publication readback mismatch')
    return value
