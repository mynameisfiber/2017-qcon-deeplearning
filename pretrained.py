import mmh3
from keras.models import load_model
from os import path
import pickle
from functools import wraps
from contextlib import redirect_stdout
import io
import copy
import sys

LOAD_PRETRAINED = False
IGNORE_TRAIN = False


class StringIOTee(io.StringIO):
    _stdout = None
    def write(self, body):
        assert self._stdout
        self._stdout.write(body)
        return super().write(body)
    
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        


def model_hash(model):
    return "model_" + str(mmh3.hash(model.to_yaml()) & 0xffffffff)

def try_pretrained_fit(model):
    model._real_fit = model.fit
    @wraps(model._real_fit)
    def _(*args, **kwargs):
        model_path = path.join('data/pretrained_models', model.mhash)
        if IGNORE_TRAIN and getattr(model, 'pretrained', False):
            with open(model_path + "_history.pkl", 'rb') as fd:
                history_out, history = pickle.load(fd)
                history._set_model(model)
            print(history_out)
        else:
            with StringIOTee() as history_out:
                history = model._real_fit(*args, **kwargs)
            print("Saving model {} to {}".format(model.mhash, model_path))
            with open(model_path + "_history.pkl", 'wb+') as fd:
                history_pkl = copy.copy(history)
                history_pkl.model = None
                pickle.dump((history_out.getvalue(), history_pkl), fd)
            model.save(model_path)
        return history
    return _

def pretrained(fxn):
    @wraps(fxn)
    def _(*args, **kwargs):
        model = fxn(*args, **kwargs)
        mhash = model.mhash = model_hash(model)
        if LOAD_PRETRAINED:
            try:
                model_path = path.join('data/pretrained_models', model.mhash)
                model = load_model(model_path)
                model.mhash = mhash
                model.pretrained = True
                print("Loaded model {} from {}".format(model.mhash, model_path))
            except IOError:
                print("Could not find model {} at {}".format(model.mhash, model_path))
                pass
        model.fit = try_pretrained_fit(model)
        return model
    return _