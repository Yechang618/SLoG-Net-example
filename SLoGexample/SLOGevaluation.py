# 2020/02/25~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
evaluation.py Evaluation Module

Methods for evaluating the models.

evaluate: evaluate a model
evaluateSingleNode: evaluate a model that has a single node forward
evaluateFlocking: evaluate a model using the flocking cost
"""

import os
import torch
import pickle

def evaluate(model, data, **kwargs):
    """
    evaluate: evaluate a model using classification error
    
    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method.
        doPrint (optional, bool): if True prints results
    
    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """

    # Get the device we're working on
    device = model.device
    
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = False
        
    if 'topN' in kwargs.keys():
        topN = kwargs['topN']
    else:
        topN = None       

    ########
    # DATA #
    ########

    Y_test,X_test,targets_test, g_test = data.get_test_Samples()

    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    with torch.no_grad():
        # Process the samples
#         print(Y_test.shape)
        xHatTest, gHatTest = model.archit(Y_test)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate_slog(xHatTest, targets_test, topN)

    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        xHatTest, gHatTest = model.archit(Y_test)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast= data.evaluate_slog(xHatTest, targets_test,topN)

    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    print(evalVars)
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars

