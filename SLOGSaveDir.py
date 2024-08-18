import os


def get_save_settings(location, **kwargs):
    thisFilename_SLOG = 'sourceLocSLOGNET'
    saveDirRoot = 'experiments' # Relative location where to save the file
    saveDir = os.path.join(saveDirRoot, thisFilename_SLOG) # Dir where to save all the results from each run
    saveSettings = {}
    saveSettings['thisFilename_SLOG'] = thisFilename_SLOG         
    saveSettings['saveDirRoot'] = saveDirRoot # Relative location where to save the file
    saveSettings['saveDir'] = os.path.join(saveDirRoot, thisFilename_SLOG) # Dir where to save all the results from each run
#     saveDirRoot_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineresults"

    if location == 'office':
#         saveDir_dropbox = '/Users/changye/Dropbox/onlineResults/experiments'
        saveDir_dropbox_root = '/Users/changye/Dropbox/onlineResults'
    else:
        saveDir_dropbox_root = r"C:\Users\Chang Ye\Dropbox\onlineResults"
#         saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
    
    saveDir_dropbox = os.path.join(saveDir_dropbox_root,'experiments')
    saveSettings['saveDirRoot_dropbox'] = saveDir_dropbox_root
    saveSettings['saveDir_dropbox'] = os.path.join(saveDir_dropbox_root, saveDir)
    
    result = {}
    result['saveDir'] = saveSettings['saveDir']
    result['saveDirRoot'] = saveSettings['saveDirRoot']   
    result['thisFilename_SLOG'] = saveSettings['thisFilename_SLOG'] 
    result['saveSettings'] = saveSettings 
    result['saveDir_dropbox'] = saveDir_dropbox
    result['saveDir_dropbox_root'] = saveDir_dropbox_root
    
    return result