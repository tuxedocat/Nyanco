# Get diff from original/correction pair sentences
# def _decode_diff(_o, _a, dt):
    # o_i1 = dt[1]; o_i2 = dt[2]
    # a_i1 = dt[3]; a_i2 = dt[4]
    # return _o[o_i1:o_i2], _a[a_i1:a_i2]

# def _decode_diff_id(_o, _a, dt):
    # o_i1 = dt[1]; o_i2 = dt[2]
    # a_i1 = dt[3]; a_i2 = dt[4]
    # return ((o_i1, o_i2), (a_i1, a_i2))

def _decode_diff(_o, _a, dt):
    o_i1 = dt[1]; o_i2 = dt[2]
    a_i1 = dt[3]; a_i2 = dt[4]
    O = tuple(_o[o_i1:o_i2])
    A = tuple(_a[a_i1:a_i2])
    return O, A, (o_i1, o_i2), (a_i1, a_i2)


def _tdiff(_o, _a):
    replaced = [t for t in difflib.SequenceMatcher(None, _o, _a).get_opcodes() if t[0] == 'replace']
    if replaced:
        tmp = [_decode_diff(_o, _a, dt) for dt in replaced]
        # print tmp
        for tt in tmp:
            # print tt
            o = tt[0]
            a = tt[1]
            ido = tt[2]
            ida = tt[3]
            # _res = [t for t in zip(o,a,ido,ida) if "VB" in t[0][1] and "VB" in t[1][1] and t[0][1]==t[1][1]]
            _res = [t for t in zip(o,a,ido,ida) if "VB" in t[0][1] and "VB" in t[1][1]]
            return _res
            # _res = (t for t in zip(o,a,ido,ida) if "VB" in t[0][1] and "VB" in t[1][1] and t[0][1]==t[1][1])
            # _res = (t for t in zip(o,a) if "VB" in t[0][1] and "VB" in t[1][1] and t[0][1]==t[1][1])
            # _res = [t for t in zip(o,a) if "VB" in t[0][1] and "VB" in t[1][1] and t[0][1]==t[1][1]]
            # _res = [t for t in zip(o,a) if "VB" in t[0][1] and "VB" in t[1][1]]
            # for r in _res:
                # print r
                # return r
            # if _res:
                # print _res
                # for r in _res:
                    # return r
            # print  [(t[0][0], t[1][0]) for t in tmp if "VB" in t[0][1] and "VB" in t[1][1]]


def get_diff(_o, _a):
    if hasattr(_o, "upper"):
        _o = _o.split()
    if hasattr(_a, "upper"):
        _a = _a.split()
    isfound = False
    while isfound is False:
        _tdiff(_o, _a)

def get_vlxc_corpus(orgd={}, corrd={}, l8db=None):
    vlxcc = defaultdict(dict)
    for k in orgd:
        o = orgd[k]
        c = corrd[k]
        pairs = [t for t in _tdiff(o,c)] if _tdiff(o,c) else None
        print pairs
        if pairs:
            for i, p in enumerate(pairs):
                print p
                _k = k+"_%d"%i
                _incorr = p[0][0]
                _corr = p[1][0]
                # _raworg = l8db[k]["original"]
                # _rawcorr = l8db[k]["correct"]
                _rawincorr = " ".join([t[0] for t in o]) 
                _rawcorr = " ".join([t[0] for t in c]) 
                vlxcc[k]["vidx_incorr"] = p[2]
                vlxcc[k]["vidx_corr"] = p[3]
                vlxcc[k]["incorr"] = _incorr
                vlxcc[k]["corr"] = _corr
                vlxcc[k]["raw_incorr"] = _rawincorr
                vlxcc[k]["raw_corr"] = _rawcorr
    return vlxcc


def add_parse(l8c, parser):
    for k in l8c:
        p = l8c[k]["raw_corr"] 
        _t = parser.parseSentence(p.encode("utf-8"))
        if _t:
            l8c[k]["parsed_corr"] = _t
        print "Done %s"%k


def _parse(s, parser):
    res = parser.parseSentence(s.encode("utf-8"))
    return res if res else [] 

def get_goldcorpus(l8corpus, cs, parser=None):
    out = defaultdict(list)
    for k ,s in cs.iteritems():
        _set = [w[0] for w in s]
        _t = [d for d in l8corpus.itervalues() if d["label_corr"] in _set and d["label_incorr"] == k ] 
        if _t:
            for t in _t:
                t["parsed_corr"] = _parse(t["raw_corr"], parser)
                out[k].append(t)
    return out


