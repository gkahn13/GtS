import pickle
import gzip
import threading

def dump(object, filename, protocol=0, compresslevel=1, async=False):
    """Saves a compressed object to disk
    """
    def run():
        file = gzip.GzipFile(filename, 'wb', compresslevel=compresslevel)
        pickle_dump = pickle.dumps(object, protocol=protocol)
        file.write(pickle_dump)
        file.close()

    if async:
        threading.Thread(target=run).start()
    else:
        run()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = b''
    while True:
        data = file.read()
        if data == b'':
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object
