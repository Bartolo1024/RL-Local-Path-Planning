import os
import datetime

def get_new_session_id():
    sess_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists('out'):
        os.makedirs('out')
    sess_dir = 'out/{}'.format(sess_id)
    if not os.path.exists(sess_dir):
        os.mkdir(sess_dir)
    return sess_id
