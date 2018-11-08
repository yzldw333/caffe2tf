
def load_pickle(pklname,sess):
    import pickle
    f = open(pklname,'rb')
    dct = pickle.load(f)
    for var in tf.global_variables():
        if var.name in dct:
            if var.shape!=dct[var.name].shape:
                if len(var.shape)==3 and len(dct[var.name].shape)==1:
                    # This is prelu, tf needs h*w*c data.
                    tmp = np.ones([var.shape[0],var.shape[1],var.shape[2]])*dct[var.name].reshape([1,1,dct[var.name].shape[0]])
                    var.load(tmp,sess)
                    continue
                else:
                    print(var.name,'not load')
            var.load(dct[var.name],sess)
