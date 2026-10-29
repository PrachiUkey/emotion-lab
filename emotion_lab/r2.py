"""R2 access. Credentials come from environment variables or the gitignored .env -- never from code."""
import configparser
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DUMP_BUCKET = 'dump-bucket'
CKPT_PREFIX = 'emotions_ckpt'
LAB_PREFIX = 'emotion-lab'


@lru_cache()
def credentials():
    """{'endpoint','access_key_id','secret_access_key'} from R2_* env vars, else the [r2] block of .env."""
    env = {k: os.environ.get(f'R2_{k.upper()}') for k in ('endpoint', 'access_key_id', 'secret_access_key')}
    if all(env.values()):
        return env
    path = Path(os.environ.get('EMOTION_LAB_ENV', REPO / '.env'))
    if not path.is_file():
        raise RuntimeError(f'No R2 credentials: set R2_ENDPOINT/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY or create {path} '
                           f'(see README, "Credentials")')
    text = path.read_text()
    start = text.find('[r2]')
    if start < 0:
        raise RuntimeError(f'{path} has no [r2] section (see README, "Credentials")')
    cp = configparser.ConfigParser()
    cp.read_string(text[start:])
    sec = cp['r2']
    creds = {k: sec.get(k, '').strip() for k in ('endpoint', 'access_key_id', 'secret_access_key')}
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise RuntimeError(f'{path} [r2] is missing {missing}')
    return creds


def s3_client():
    import boto3
    from botocore.config import Config
    c = credentials()
    return boto3.client('s3', endpoint_url=c['endpoint'], aws_access_key_id=c['access_key_id'],
                        aws_secret_access_key=c['secret_access_key'], region_name='auto',
                        config=Config(retries={'max_attempts': 8, 'mode': 'adaptive'}))


def lance_storage_options():
    c = credentials()
    return {'endpoint': c['endpoint'], 'access_key_id': c['access_key_id'],
            'secret_access_key': c['secret_access_key'], 'region': 'auto',
            'virtual_hosted_style_request': 'false'}


def rclone_env():
    """Environment that defines an rclone remote named `r2` without writing a config file."""
    c = credentials()
    return {**os.environ, 'RCLONE_CONFIG_R2_TYPE': 's3', 'RCLONE_CONFIG_R2_PROVIDER': 'Cloudflare',
            'RCLONE_CONFIG_R2_ENDPOINT': c['endpoint'], 'RCLONE_CONFIG_R2_ACCESS_KEY_ID': c['access_key_id'],
            'RCLONE_CONFIG_R2_SECRET_ACCESS_KEY': c['secret_access_key'], 'RCLONE_CONFIG_R2_REGION': 'auto',
            'RCLONE_CONFIG_R2_NO_CHECK_BUCKET': 'true'}
