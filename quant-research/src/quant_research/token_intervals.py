"""Date-weighted empirical interval calibration; no iid coverage guarantee."""
import numpy as np


def validate_quantiles(q):
    q=np.asarray(q,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():
        raise ValueError('Expected finite lower, median and upper quantiles')
    if (np.diff(q,axis=1)<0).any():
        raise ValueError('Quantiles must be ordered')
    return q


def fit_scale(q,y,dates,coverage=.8,epsilon=1e-6):
    q=validate_quantiles(q)
    y=np.asarray(y,dtype=np.float64)
    dates=np.asarray(dates)
    if y.shape!=(len(q),) or dates.shape!=y.shape or not np.isfinite(y).all() or not len(q):
        raise ValueError('Calibration labels and dates must align')
    if not 0<coverage<1 or epsilon<=0:
        raise ValueError('Invalid calibration settings')
    _,code,counts=np.unique(dates,return_inverse=True,return_counts=True)
    weight=1./counts[code]
    left=np.maximum(q[:,1]-q[:,0],epsilon)
    right=np.maximum(q[:,2]-q[:,1],epsilon)
    score=np.maximum((q[:,1]-y)/left,(y-q[:,1])/right)
    order=np.argsort(score,kind='stable')
    cdf=np.cumsum(weight[order])/weight.sum()
    index=min(np.searchsorted(cdf,coverage),len(order)-1)
    scale=max(1.,float(score[order[index]]))
    if not np.isfinite(scale):
        raise ValueError('Nonfinite calibration scale')
    return dict(scale=scale,rows=len(q),dates=len(counts),coverage=coverage,epsilon=epsilon,
        degenerate_halfwidths=int(((q[:,1]-q[:,0]<epsilon)|(q[:,2]-q[:,1]<epsilon)).sum()))


def apply_scale(q,scale,lower_support,epsilon=1e-6):
    q=validate_quantiles(q)
    if not np.isfinite(scale) or scale<1 or epsilon<=0 or not np.isfinite(lower_support):
        raise ValueError('Invalid interval calibration')
    if (q[:,0]<lower_support).any():
        raise ValueError('Raw quantiles violate target support')
    out=q.copy()
    out[:,0]=np.maximum(lower_support,q[:,1]-scale*np.maximum(q[:,1]-q[:,0],epsilon))
    out[:,2]=q[:,1]+scale*np.maximum(q[:,2]-q[:,1],epsilon)
    return out


def interval_metrics(q,y):
    q=validate_quantiles(q)
    y=np.asarray(y,dtype=np.float64)
    if y.shape!=(len(q),) or not np.isfinite(y).all():
        raise ValueError('Invalid interval labels')
    lower,median,upper=q.T
    return dict(mae=np.abs(median-y),coverage80=((lower<=y)&(y<=upper)).astype(float),
        width80=upper-lower,interval_score80=upper-lower+10*np.maximum(lower-y,0)+10*np.maximum(y-upper,0),
        lower_miss=(y<lower).astype(float),upper_miss=(y>upper).astype(float))
