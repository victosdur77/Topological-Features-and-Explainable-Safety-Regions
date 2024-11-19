import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay


from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.lines as mlines

def plot_SSVM(Xtr, Ytr, Xts, Yts, model2d, dimGrid = 100, featurelabels = ["X1","X2"], plot_method = "contourf", transparency = 0.3, colormap = "bwr", mycolor = None):

    """ plot a 2d scatter with (scalable) SVM decision boundary 
    Xtr: 2D input training data
    Ytr: training real labels
    Xts: 2D test data
    Yts: test real labels
    model2D: ScalableSVMClassifier model trained on 2D data
    
    """
    
    # define meshgrid ranges based on the input features
    
    x_range = np.linspace(min(Xtr[:,0])-1,max(Xtr[:,0])+1,dimGrid)
    y_range = np.linspace(min(Xtr[:,0])-1,max(Xtr[:,0])+1,dimGrid)
    
    [XX, YY] = np.meshgrid(x_range, y_range)

    E = np.vstack((XX.flatten(), YY.flatten())).T
    
    if model2d.kernel == "linear":
        w = -model2d.eta*(model2d.alpha.T @ np.diag(Ytr)) @ Xtr
        
        f = lambda z: 1/w[1]*((model2d.b - model2d.b_eps) - w[0]*z)

        y_lin = f(x_range)

        hyperplane_func = lambda t: w @ t.T - model2d.b + model2d.b_eps


        func_values = hyperplane_func(E)

        func_values_matrix = np.reshape(func_values, (len(x_range), len(y_range)))
        
        
        filled_region = func_values_matrix[func_values_matrix <= 0]

        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.contourf(x_range, y_range, filled_region, [1, 1], facecolor = "blue", alpha = 0.2);
        ax.plot(x_range,y_lin, c="blue", linestyle='dashed',linewidth=2)

    else:
        
        y_pred = model2d.predict(E)

        fig = plt.figure()
        ax = fig.add_subplot()

        if mycolor!=None:
            mycolormap = [mycolor[0]*np.ones((256,1)), mycolor[1]*np.ones((256,1)), mycolor[3]*np.ones((256,1))]
        else:
            mycolormap = colormap
        # np.reshape(y_pred,(len(YY),len(XX)))
        ax.contourf(XX, YY, np.reshape(y_pred,XX.shape) ,[0,1],alpha = 0.3, cmap = mycolormap)
        s = ax.contour(XX, YY, np.reshape(y_pred,XX.shape) ,[0,1], colors = "b")
        
        sc1 = ax.scatter(Xts[:, 0][Yts==model2d.classes_[0]], Xts[:, 1][Yts==model2d.classes_[0]], c="red",s=15, edgecolor="k", label = str(int(model2d.classes_[0])) )
        sc2 = ax.scatter(Xts[:, 0][Yts==model2d.classes_[1]], Xts[:, 1][Yts==model2d.classes_[1]], c="limegreen",s=15, edgecolor="k", label = str(int(model2d.classes_[1])) )

        h = s.collections
        l = [f'{s.levels[0]:.1f}']

        blue_line = mlines.Line2D([], [], color='blue',markersize=15, label='SafetyRegion')
    ax.legend(handles = [sc1,sc2,blue_line])
    plt.show()
    
    '''
    # OTHERWISE, AUTOMATICALLY FROM THE ESTIMATOR
    disp = DecisionBoundaryDisplay.from_estimator(model2d,Xts,response_method="predict",xlabel=featurelabels[0], ylabel=featurelabels[1],alpha=transparency, cmap = colormap, plot_method = plot_method, edgecolor = "k", label = "SVM boudary")

    disp.ax_.scatter(Xts[:, 0][Yts==model2d.classes_[0]], Xts[:, 1][Yts==model2d.classes_[0]], c="red",s=25, edgecolor="k", label = str(int(model2d.classes_[0])) )
    disp.ax_.scatter(Xts[:, 0][Yts==model2d.classes_[1]], Xts[:, 1][Yts==model2d.classes_[1]], c="limegreen", s= 25, edgecolor="k", label = str(int(model2d.classes_[1])) )
    #plt.colorbar(disp, label = "SC-SVM Boundary", shrink = 0.5)
    if plot_method != "contour":
        plt.colorbar(disp.surface_,label = "SC-SVM Boundary")
    disp.ax_.legend(loc = "upper left")
    plt.show()
    '''