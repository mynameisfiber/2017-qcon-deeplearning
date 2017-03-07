# ML Workshop Files

![](images/easyinstall.gif)

To install:

```bash
$ pip3 install -U keras jupyter matplotlib seaborn sklearn
```

Next, make sure that keras is properly configured. The `~/.keras/keras.json`
file should have the `backend = theano` and `image_dim_ordering = th` and look
like,

```bash
$ cat ~/.keras/keras.json
{
  "epsilon": 1e-07,
  "floatx": "float32",
  "backend": "theano",
  "image_dim_ordering": "th"
}
```

## Slides

- [micha.codes/2017-qcon-deeplearning](http://micha.codes/2017-qcon-deeplearning)

## Notebooks

- [MNIST.ipynb](MNIST.ipynb) is the notebook I ran through without running code
  in the second half.

- [failed_workshop_code.ipynb](failed_workshop_code.ipynb) is the aborted
  notebook I put together live before my browser crashed.
