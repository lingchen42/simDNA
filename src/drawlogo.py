import numpy as np
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=fp) }
COLOR_SCHEME = {'G': 'orange',
                'A': 'crimson',
                'C': 'mediumblue',
                'T': 'forestgreen'}

#class Scale(matplotlib.patheffects.RendererBase):
#    def __init__(self, sx, sy=None):
#        self._sx = sx
#        self._sy = sy
#
#    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
#        affine = affine.identity().scale(self._sx, self._sy)+affine
#        renderer.draw_path(gc, tpath, affine, rgbFace)


def motif2scores(motif, bg=[0.25, 0.25, 0.25, 0.25],
                 smoothing_constant=0.001):
    ''' convert the 4 x n motif to score format like below
    '''
#     ALL_SCORES1 = [[('C', 0.02247014831444764),
#               ('T', 0.057903843733384308),
#               ('A', 0.10370837683591219),
#               ('G', 0.24803586793255664)],
#               [('T', 0.046608227674354567),
#               ('G', 0.048827667087419063),
#               ('A', 0.084338697696451109),
#               ('C', 0.92994511407402669)]]
    base_d = {0:'A', 1:'C', 2:'G', 3:'T'}

    # column normalize the motif
    m = motif.copy()
    m += smoothing_constant
    m = m / m.sum(axis=0)

    # scale it by information content
    m = m * (m * np.log2(m.T / np.array(bg)).T).sum(axis=0)

    scores = []
    m_len = np.shape(m)[1]
    for i in range(m_len):
        column_score = [(base_d[l], m[l, i]) for l in np.argsort(m[:, i])]
        scores.append(column_score)
    return scores


def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


def draw_logo(motif):
    all_scores = motif2scores(motif)

    fig, ax = plt.subplots(figsize=(len(all_scores), 3))
    x = 1
    for scores in all_scores:
        y = 0
        for base, score in scores:
            letterAt(base, x,y, score, ax)
            y += score
        x += 1

    plt.xticks(range(1,x))
    plt.xlim((0, x))
    plt.ylim((0, 2))
    plt.tight_layout()

    return fig


#def draw_logo(motif, fontfamily='Arial', size=80):
#    all_scores = motif2scores(motif)
#    
#    if fontfamily == 'xkcd':
#        plt.xkcd()
#    else:
#        mpl.rcParams['font.family'] = fontfamily
#
#    fig, ax = plt.subplots(figsize=(len(all_scores), 2.5))
#
#    font = FontProperties()
#    font.set_size(size)
#    font.set_weight('bold')
#    
#    #font.set_family(fontfamily)
#
#    ax.set_xticks(range(1,len(all_scores)+1))    
#    ax.set_yticks(range(0,3))
#    ax.set_xticklabels(range(1,len(all_scores)+1), rotation=90)
#    ax.set_yticklabels(np.arange(0,3,1))    
#    seaborn.despine(ax=ax, trim=True)
#    
#    trans_offset = transforms.offset_copy(ax.transData, 
#                                          fig=fig, 
#                                          x=1, 
#                                          y=0, 
#                                          units='dots')
#   
#    for index, scores in enumerate(all_scores):
#        yshift = 0
#        for base, score in scores:
#            txt = ax.text(index+1, 
#                          0, 
#                          base, 
#                          transform=trans_offset,
#                          fontsize=80, 
#                          color=COLOR_SCHEME[base],
#                          ha='center',
#                          fontproperties=font,
#
#                         )
#            txt.set_path_effects([Scale(1.0, score)])
#            fig.canvas.draw()
#            window_ext = txt.get_window_extent(txt._renderer)
#            yshift = window_ext.height*score
#            trans_offset = transforms.offset_copy(txt._transform, 
#                                                  fig=fig,
#                                                  y=yshift,
#                                                  units='points')
#        trans_offset = transforms.offset_copy(ax.transData, 
#                                              fig=fig, 
#                                              x=1, 
#                                              y=0, 
#                                              units='points')    
#    return fig, ax
