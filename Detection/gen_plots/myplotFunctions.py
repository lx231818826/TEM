
import numpy, string
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------------------------------------------------
def myPlotPrecRecLOGDetector(ROC_precision,ROC_recall,styles,\
                             method, logLegendsRadius, logLegendsRadiusStep, dataset, withcv, \
                             fig = None, ax = None, leg=None, counter=0,
                             lim = None, last=False):

        
	count = counter
        # print count

        if fig is None:
                fig = plt.figure(figsize=(20,10),dpi=100)
                ax  = fig.add_subplot(111)

        (nPoints,dump) = ROC_precision.shape
        
        if withcv or leg!=None:
                nRadius = 1
        else:
                (nRadius,) = logLegendsRadius.shape

	for nRd in xrange(0,nRadius):
                # legnamesFinal.append(legname)
                if withcv:
                        # count   = 5
                        legname = leg
                        prec = ROC_precision[:,0]
                        rec  = ROC_recall[:,0]
                        linestyle = ""

                elif leg!=None:
                        # count   = 6
                        legname = leg
                        prec = ROC_precision[:,0]
                        rec  = ROC_recall[:,0]
                        linestyle = styles['plinestyle'][0]

                else:
                        legname = 'Radius={0:02d}'.format(logLegendsRadius[nRd])
                        prec = ROC_precision[:,nRd]
                        rec  = ROC_recall[:,nRd]
                        linestyle = styles['plinestyle'][0]

                print legname
                for j in range(0,len(prec)):
                        print "Precision: {0:05f} | Recall: {1:05f} ".format(prec[j],rec[j])
                                
                ax.plot(prec, rec,
                        color           = styles['pcolors'][count],
                        marker          = styles['pmarkers'][count],
                        markerfacecolor = styles['pcolors'][count],
                        markeredgecolor = styles['pedgecolor'][0],
                        markeredgewidth = 1,
                        linestyle       = linestyle,
                        linewidth       = 4,
                        markersize      = 35,
                        label 	        = legname,
                        zorder          = 5)
                

                count = count + 1

                if count > len( styles['pcolors'] )-1:
                        count = 0

                ax.set_xlim((0,1.))
                ax.set_ylim((0,1.))

	handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,-0.2),
                        numpoints = 1,
                        ncol = 3, prop={'size': styles['fontsize']})

	# leg = legend( legnamesFinal, 3)
        # -------------------------
	# best          0
        # upper right   1
        # upper left    2
        # lower left    3
        # lower right   4
        # right         5
        # center left   6
        # center right  7
        # lower center  8
        # upper center  9
        # center        10
        # -------------------------
        ax.grid(alpha=0.7,linewidth=2,zorder=0)

        for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(styles['fontsize']-2)
        for tick in ax.yaxis.get_major_ticks():
        	tick.label1.set_fontsize(styles['fontsize']-2)
        #for t in leg.get_texts():
        #	t.set_fontsize(styles['fontsize'])

	# params = {'font.family'    : 'serif',
        #           #'font.serif'     : 'New Century Schoolbook'
        # }
	# rcdefaults()
        # rcParams.update(params)


        #method2 = method.replace('log_detector','LoG detector')
	mtitlef = method.replace('_',' ')
        #print mtitlef
        #kakakak
        plt.title(mtitlef, fontsize=styles['fontsize']+15, verticalalignment='bottom')
        ax.set_xlabel('Precision', fontsize=styles['fontsize']+5)
        ax.set_ylabel('Recall', fontsize=styles['fontsize']+5)


        # plot radius last 2 threshold values
        if lim == None:
                xmin, xmax = (1,0)
                ymin, ymax = (1,0)
        else:
                (xmin, xmax, ymin, ymax) = lim
                
	count = counter
        ax2   = plt.axes([.2, .2, .3, .3], axisbg='w')
        ax2.set_title("")
        for nRd in xrange(0,nRadius):
                prec = ROC_precision[:,nRd]
                nkeypoints = len(prec)

                npoints = 3
                if nRd != 0:
                        npoints = 2
                        
                
                prec = ROC_precision[nkeypoints-npoints:,nRd]
                rec  = ROC_recall[nkeypoints-npoints:,nRd]

                if nRd != nRadius-1 or withcv:
                        xmin = min(numpy.min(prec),xmin)
                        xmax = max(numpy.max(prec),xmax)
                        ymin = min(numpy.min(rec),ymin)
                        ymax = max(numpy.max(rec),ymax)
                
                        ax2.plot(prec,rec,
                                 color           = styles['pcolors'][count],
                                 marker          = styles['pmarkers'][count],
                                 markerfacecolor = styles['pcolors'][count],
                                 markeredgecolor = styles['pedgecolor'][0],
                                 markeredgewidth = 1,
                                 linestyle       = styles['plinestyle'][0],
                                 linewidth       = 4,
                                 markersize      = 30,
                                 label 	         = legname,
                                 zorder          = 5)
                
                plt.setp(ax2)
                
                count = count + 1
                
                if count > len( styles['pcolors'] )-1:
                        count = 0

        #if not withcv:
        # print xmin, xmax, ymin, ymax
        ax2.set_xlim((xmin-.01,xmax+.01))
        ax2.set_ylim((ymin-.01,ymax+.01))


        if last:
                xmin = xmin-.1
                ymin = ymin-.1
                xmax = xmax+.1
                ymax = ymax+.1
                ax.add_patch(
                        patches.Rectangle(
                                (xmin,ymin),
                                xmax-xmin,
                                ymax-ymin,
                                fill=False,
                                linewidth=3
                        )
                )

        
	savefigname = 'imgs/' + method[4:].replace(' ','_') + '_' + dataset + '.svg';
        print 'Saving figure in: ' + savefigname
        fig.savefig(savefigname,format='svg',dpi=200,
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

        limits = (xmin,xmax,ymin,ymax)
        return (fig,ax,count, limits)
