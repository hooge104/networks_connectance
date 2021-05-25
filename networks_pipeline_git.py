# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import tqdm
import time
import datetime
from pathlib import Path
import ee
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from itertools import combinations

ee.Initialize()

classProperty = 'connectance_windsor'

def assessExtrapolation(importedData, compositeImage, propOfVariance):

    # Excise the columns of interest from the data frame
    variablesOfInterest = importedData

    # Compute the mean and standard deviation of each band, then standardize the point data
    meanVector = variablesOfInterest.mean()
    stdVector = variablesOfInterest.std()
    standardizedData = (variablesOfInterest-meanVector)/stdVector

    # Then standardize the composite from which the points were sampled
    meanList = meanVector.tolist()
    stdList = stdVector.tolist()
    bandNames = list(meanVector.index)
    meanImage = ee.Image(meanList).rename(bandNames)
    stdImage = ee.Image(stdList).rename(bandNames)
    standardizedImage = compositeImage.subtract(meanImage).divide(stdImage)

    # Run a PCA on the point samples
    pcaOutput = PCA()
    pcaOutput.fit(standardizedData)

    # Save the cumulative variance represented by each PC
    cumulativeVariance = np.cumsum(np.round(pcaOutput.explained_variance_ratio_, decimals=4)*100)

    # Make a list of PC names for future organizational purposes
    pcNames = ['PC'+str(x) for x in range(1,variablesOfInterest.shape[1]+1)]

    # Get the PC loadings as a data frame
    loadingsDF = pd.DataFrame(pcaOutput.components_,columns=[str(x)+'_Loads' for x in bandNames],index=pcNames)

    # Get the original data transformed into PC space
    transformedData = pd.DataFrame(pcaOutput.fit_transform(standardizedData,standardizedData),columns=pcNames)

    # Make principal components images, multiplying the standardized image by each of the eigenvectors
    # Collect each one of the images in a single image collection;

    # First step: make an image collection wherein each image is a PC loadings image
    listOfLoadings = ee.List(loadingsDF.values.tolist());
    eePCNames = ee.List(pcNames)
    zippedList = eePCNames.zip(listOfLoadings)
    def makeLoadingsImage(zippedValue):
        return ee.Image.constant(ee.List(zippedValue).get(1)).rename(bandNames).set('PC',ee.List(zippedValue).get(0))
    loadingsImageCollection = ee.ImageCollection(zippedList.map(makeLoadingsImage))

    # Second step: multiply each of the loadings image by the standardized image and reduce it using a "sum"
    # to finalize the matrix multiplication
    def finalizePCImages(loadingsImage):
        return ee.Image(loadingsImage).multiply(standardizedImage).reduce('sum').rename([ee.String(ee.Image(loadingsImage).get('PC'))]).set('PC',ee.String(ee.Image(loadingsImage).get('PC')))
    principalComponentsImages = loadingsImageCollection.map(finalizePCImages)

    # Choose how many principal components are of interest in this analysis based on amount of
    # variance explained
    numberOfComponents = sum(i < propOfVariance for i in cumulativeVariance)+1
    print('Number of Principal Components being used:',numberOfComponents)

    # Compute the combinations of the principal components being used to compute the 2-D convex hulls
    tupleCombinations = list(combinations(list(pcNames[0:numberOfComponents]),2))
    print('Number of Combinations being used:',len(tupleCombinations))

    # Generate convex hulls for an example of the principal components of interest
    cHullCoordsList = list()
    for c in tupleCombinations:
        firstPC = c[0]
        secondPC = c[1]
        outputCHull = ConvexHull(transformedData[[firstPC,secondPC]])
        listOfCoordinates = transformedData.loc[outputCHull.vertices][[firstPC,secondPC]].values.tolist()
        flattenedList = [val for sublist in listOfCoordinates for val in sublist]
        cHullCoordsList.append(flattenedList)

    # Reformat the image collection to an image with band names that can be selected programmatically
    pcImage = principalComponentsImages.toBands().rename(pcNames)

    # Generate an image collection with each PC selected with it's matching PC
    listOfPCs = ee.List(tupleCombinations)
    listOfCHullCoords = ee.List(cHullCoordsList)
    zippedListPCsAndCHulls = listOfPCs.zip(listOfCHullCoords)

    def makeToClassifyImages(zippedListPCsAndCHulls):
        imageToClassify = pcImage.select(ee.List(zippedListPCsAndCHulls).get(0)).set('CHullCoords',ee.List(zippedListPCsAndCHulls).get(1))
        classifiedImage = imageToClassify.rename('u','v').classify(ee.Classifier.spectralRegion([imageToClassify.get('CHullCoords')]))
        return classifiedImage

    classifedImages = ee.ImageCollection(zippedListPCsAndCHulls.map(makeToClassifyImages))
    finalImageToExport = classifedImages.sum().divide(ee.Image.constant(len(tupleCombinations)))

    return finalImageToExport

# Input the proportion of variance that you would like to cover when running the script
propOfVariance = 90


# Configuration
####################################################################################
# Input the name of the username that serves as the home folder for asset storage
usernameFolderString = ''

# Input the normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 5

# Input a longer wait time (in seconds) for "wait and break" cells
longWaitTime = 10

# Specify the column names where the latitude and longitude information is stored
latString = 'Pixel_Lat'
longString = 'Pixel_Long'

# Input the name of the property that holds the CV fold assignment
cvFoldString = 'CV_Fold'

# Set k for k-fold CV
k = 10

# Make a list of the k-fold CV assignments to use
kList = list(range(1,k+1))

# Set number of trees in RF models
nTrees = 300

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = classProperty+"CV_Fold_Collection"

# Write the name of a local staging area folder for outputted CSV's
holdingFolder = 'data/training_data/'+classProperty

# Create directory to hold training data
Path(holdingFolder).mkdir(parents=True, exist_ok=True)

####################################################################################
# Image export settings
# Set pyramidingPolicy for exporting purposes
pyramidingPolicy = 'mean'

# Load a geometry to use for the export
exportingGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False);

####################################################################################
# Bootstrapping inputs
# Number of bootstrap iterations
bootstrapIterations = 100

# Generate the seeds for bootstrapping
seedsToUseForBootstrapping = list(range(1, bootstrapIterations+1))

# Input the name of a folder used to hold the bootstrap collections
bootstrapCollFolder = 'Bootstrap_Collections'

# Input the header text that will name each bootstrapped dataset
fileNameHeader = classProperty+'BootstrapColl_'

# Stratification inputs
# Write the name of the variable used for stratification
stratificationVariableString = "Resolve_Biome"

# Input the dictionary of values for each of the stratification category levels
strataDict = {
    1: 14.900835665820974,
    2: 2.941697660221864,
    3: 0.526059731441294,
    4: 9.56387696566245,
    5: 2.865354077500338,
    6: 11.519674266872787,
    7: 16.26999434439293,
    8: 8.047078485979089,
    9: 0.861212221078014,
    10: 3.623974712557433,
    11: 6.063922959332467,
    12: 2.5132866428302836,
    13: 20.037841544639985,
    14: 0.26519072167008,
}

####################################################################################
# Bash and Google Cloud Bucket settings
# Specify the necessary arguments to upload the files to a Cloud Storage bucket
# I.e., create bash variables in order to create/check/delete Earth Engine Assets

# Specify bash arguments
arglist_preEEUploadTable = ['upload','table']
arglist_postEEUploadTable = ['--x_column', longString, '--y_column', latString]
arglist_preGSUtilUploadFile = ['cp']
formattedBucketOI = 'gs://'+bucketOfInterest
assetIDStringPrefix = '--asset_id='
arglist_CreateCollection = ['create','collection']
arglist_CreateFolder = ['create','folder']
arglist_Detect = ['asset','info']
arglist_Delete = ['rm','-r']
stringsOfInterest = ['Asset does not exist or is not accessible']

# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_Delete = [bashFunction_EarthEngine]+arglist_Delete
bashCommandList_CreateCollection = [bashFunction_EarthEngine]+arglist_CreateCollection
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder

####################################################################################
# Covariate input settings
# Input a list of the covariates being used
covariateList = [
'CGIAR_PET',
'CHELSA_BIO_Annual_Mean_Temperature',
'CHELSA_BIO_Annual_Precipitation',
'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
'CHELSA_BIO_Precipitation_Seasonality',
'ConsensusLandCover_Human_Development_Percentage',
'ConsensusLandCoverClass_Barren',
'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
'ConsensusLandCoverClass_Herbaceous_Vegetation',
'ConsensusLandCoverClass_Mixed_Other_Trees',
'ConsensusLandCoverClass_Shrubs',
'EarthEnvTexture_CoOfVar_EVI',
'EarthEnvTexture_Correlation_EVI',
'EarthEnvTexture_Homogeneity_EVI',
'EarthEnvTopoMed_AspectCosine',
'EarthEnvTopoMed_AspectSine',
'EarthEnvTopoMed_Elevation',
'EarthEnvTopoMed_Slope',
'EarthEnvTopoMed_TopoPositionIndex',
'EsaCci_BurntAreasProbability',
'GHS_Population_Density',
'GlobBiomass_AboveGroundBiomass',
'GlobPermafrost_PermafrostExtent',
'MODIS_NDVI',
'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
'SG_Depth_to_bedrock',
'SG_Sand_Content_005cm',
'SG_SOC_Content_005cm',
'SG_Soil_pH_H2O_005cm',
]

# Load the composite on which to perform the mapping, and subselect the bands of interest
compositeToClassify = ee.Image("path_to_composite").select(covariateList)

####################################################################################################################################################################
# Start of modeling
####################################################################################################################################################################

# Turn the folder string into an assetID and perform the folder creation
assetIDToCreate_Folder = 'users/'+usernameFolderString+'/'+projectFolder
print(assetIDToCreate_Folder,'being created...')

# Create the folder within Earth Engine
subprocess.run(bashCommandList_CreateFolder+[assetIDToCreate_Folder])
while any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest):
    print('Waiting for asset to be created...')
    time.sleep(normalWaitTime)
print('Asset created!')

# Sleep to allow the server time to receive incoming requests
time.sleep(normalWaitTime/2)

# Import the raw CSV being bootstrapped\
rawPointCollection = pd.read_csv('data/20210219_networks_sampled.csv', float_precision='round_trip')

# Print basic information on the csv
print('Original Collection', rawPointCollection.shape[0])

# Remove the "system:index" and rename the ".geo" column to "geo" and shuffle the data frame while setting a new index
# (to ensure geographic clumps of points are not clumped in anyway
# Drop NA observations in classProperty column
fcToAggregate = rawPointCollection.sample(frac=1).reset_index(drop=True)
# print(fcToAggregate.shape[0])

# Pixel aggregation
# Select variables to include
preppedCollection = pd.DataFrame(fcToAggregate.groupby(['Pixel_Lat', 'Pixel_Long']).mean().to_records())[covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']]
print('Number of aggregated pixels', preppedCollection.shape[0])

# Drop NAs
preppedCollection = preppedCollection.dropna(how='any')
print('After dropping NAs', preppedCollection.shape[0])
# print(preppedCollection.isna().sum())

# Convert biome column to int, to correct possible rounding errors
preppedCollection[stratificationVariableString] = preppedCollection[stratificationVariableString].astype(int)

# Add fold assignments to each of the points, stratified by biome
preppedCollection[cvFoldString] = (preppedCollection.groupby('Resolve_Biome').cumcount() % k) + 1

# Write the CSV to disk and upload it to Earth Engine as a Feature Collection
localPathToCVAssignedData = holdingFolder+'/'+titleOfCSVWithCVAssignments+'.csv'
preppedCollection.to_csv(localPathToCVAssignedData,index=False)


# Format the bash call to upload the file to the Google Cloud Storage bucket
gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+[localPathToCVAssignedData]+[formattedBucketOI]
subprocess.run(gsutilBashUploadList)
print(titleOfCSVWithCVAssignments+' uploaded to a GCSB!')

# Wait for a short period to ensure the command has been received by the server
time.sleep(normalWaitTime/2)

# Wait for the GSUTIL uploading process to finish before moving on
while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in [titleOfCSVWithCVAssignments]):
    print('Not everything is uploaded...')
    time.sleep(normalWaitTime)
print('Everything is uploaded; moving on...')


# Upload the file into Earth Engine as a table asset
assetIDForCVAssignedColl = 'users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments
earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIDForCVAssignedColl]+[formattedBucketOI+'/'+titleOfCSVWithCVAssignments+'.csv']+arglist_postEEUploadTable
subprocess.run(earthEngineUploadTableCommands)
print('Upload to EE queued!')

# Wait for a short period to ensure the command has been received by the server
time.sleep(normalWaitTime/2)

# !! Break and wait
count = 1
while count >= 1:
    taskList = [str(i) for i in ee.batch.Task.list()]
    subsetList = [s for s in taskList if classProperty in s]
    subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
    count = len(subsubList)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
    time.sleep(normalWaitTime)
print('Moving on...')


# remove line below when un-commenting gc upload above
assetIDForCVAssignedColl = 'users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments

# Load the collection with the pre-assigned K-Fold assignments
fcOI = ee.FeatureCollection(assetIDForCVAssignedColl)

varsPerSplit_list = list(range(1,14))
leafPop_list = list(range(2,14))
classifierList = []

for vps in varsPerSplit_list:
    for lp in leafPop_list:

        model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp)

        rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
        numberOfTrees=nTrees,
        variablesPerSplit=vps,
        minLeafPopulation=lp,
        bagFraction=0.632
        ).setOutputMode('REGRESSION'))

        classifierList.append(rf)

# Define the R^2 function
def coefficientOfDetermination(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the mean of the property of interest
    propertyOfInterestMean = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).select([propertyOfInterest]).reduceColumns(ee.Reducer.mean(),[propertyOfInterest])).get('mean'));

    # Compute the total sum of squares
    def totalSoSFunction(f):
        return f.set('Difference_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(propertyOfInterestMean).pow(ee.Number(2)))
    totalSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(totalSoSFunction).select(['Difference_Squared']).reduceColumns(ee.Reducer.sum(),['Difference_Squared'])).get('sum'))

    # Compute the residual sum of squares
    def residualSoSFunction(f):
        return f.set('Residual_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(ee.Number(ee.Feature(f).get(propertyOfInterest_Predicted))).pow(ee.Number(2)))
    residualSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(residualSoSFunction).select(['Residual_Squared']).reduceColumns(ee.Reducer.sum(),['Residual_Squared'])).get('sum'))

    # Finalize the calculation
    r2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares))

    return ee.Number(r2)

# Define the RMSE function
def RMSE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the squared difference between observed and predicted
    def propDiff(f):
        diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))

        return f.set('diff', diff.pow(2))

    # calculate RMSE from squared difference
    rmse = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean')).sqrt()

    return rmse

# Define the MAE function
def MAE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the absolute difference between observed and predicted
    def propDiff(f):
        diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))

        return f.set('diff', diff.abs())

    # calculate RMSE from squared difference
    mae = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean'))

    return mae

# Make a feature collection from the k-fold assignment list
kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

# Define a function to take a feature with a classifier of interest
def computeCVAccuracyAndRMSE(featureWithClassifier):
    # Pull the classifier from the feature
    cOI = ee.Classifier(featureWithClassifier.get('c'))

    # Create a function to map through the fold assignments and compute the overall accuracy
    # for all validation folds
    def computeAccuracyForFold(foldFeature):
        # Organize the training and validation data
        foldNumber = ee.Number(ee.Feature(foldFeature).get('Fold'))
        trainingData = fcOI.filterMetadata(cvFoldString,'not_equals',foldNumber)
        validationData = fcOI.filterMetadata(cvFoldString,'equals',foldNumber)

        # Train the classifier and classify the validation dataset
        trainedClassifier = cOI.train(trainingData,classProperty,covariateList)
        outputtedPropName = classProperty+'_Predicted'
        classifiedValidationData = validationData.classify(trainedClassifier,outputtedPropName)

        # Compute accuracy metrics
        r2ToSet = coefficientOfDetermination(classifiedValidationData,classProperty,outputtedPropName)
        rmseToSet = RMSE(classifiedValidationData,classProperty,outputtedPropName)
        maeToSet = MAE(classifiedValidationData,classProperty,outputtedPropName)
        return foldFeature.set('R2',r2ToSet).set('RMSE', rmseToSet).set('MAE', maeToSet)

    # Compute the mean and std dev of the accuracy values of the classifier across all folds
    accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)
    meanAccuracy = accuracyFC.aggregate_mean('R2')
    sdAccuracy = accuracyFC.aggregate_total_sd('R2')

    # Calculate mean and std dev of RMSE
    RMSEvals = accuracyFC.aggregate_array('RMSE')
    RMSEvalsSquared = RMSEvals.map(lambda f: ee.Number(f).multiply(f))
    sumOfRMSEvalsSquared = RMSEvalsSquared.reduce(ee.Reducer.sum())
    meanRMSE = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared).divide(k))

    sdRMSE = accuracyFC.aggregate_total_sd('RMSE')

    # Calculate mean and std dev of MAE
    meanMAE = accuracyFC.aggregate_mean('MAE')
    sdMAE= accuracyFC.aggregate_total_sd('MAE')

    # Compute the feature to return
    featureToReturn = featureWithClassifier.select(['cName']).set('Mean_R2',meanAccuracy,'StDev_R2',sdAccuracy, 'Mean_RMSE',meanRMSE,'StDev_RMSE',sdRMSE, 'Mean_MAE',meanMAE,'StDev_MAE',sdMAE)
    return featureToReturn


classDf = pd.DataFrame(columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'])

for rf in tqdm.tqdm(classifierList):

    accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))

    classDf = classDf.append(pd.DataFrame(accuracy_feature.getInfo()['properties'], index = [0]))

classDfSorted = classDf.sort_values([sort_acc_prop], ascending = False)

print('Top 5 grid search results:\n', classDfSorted.head(5))

bestModelName = classDfSorted.iloc[0]['cName']

print('Best model:', bestModelName)

# Write model results to csv
modelResults = pd.DataFrame({'time': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                             'classProperty': classProperty,
                             'bestModelName': bestModelName,
                             'Mean_R2': classDfSorted.iloc[0]['Mean_R2'],
                             'StDev_R2': classDfSorted.iloc[0]['StDev_R2'],
                             'Mean_RMSE': classDfSorted.iloc[0]['Mean_RMSE'],
                             'StDev_RMSE': classDfSorted.iloc[0]['StDev_RMSE'],
                             'Mean_MAE': classDfSorted.iloc[0]['Mean_MAE'],
                             'StDev_MAE': classDfSorted.iloc[0]['StDev_MAE']}, index = [0])

with open('model_details.csv', 'a') as f:
    modelResults.to_csv(f, mode='a', header=f.tell()==0)

# Variable importance metrics
# Load the best model from the classifier list
classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).
                                          filterMetadata('cName', 'equals', bestModelName).first()).get('c'))

# Train the classifier with the collection
trainedClassiferForSingleMap = classifier.train(fcOI, classProperty, covariateList)

# Get the feature importance from the trained classifier and print
# them to a .csv file and as a bar plot as .png file
classifierDict = trainedClassiferForSingleMap.explain().get('importance')
featureImportances = classifierDict.getInfo()
featureImportances = pd.DataFrame(featureImportances.items(),
                                  columns=['Covariates', 'Feature_Importance']).sort_values(by='Feature_Importance',
                                                                                            ascending=False)
featureImportances.to_csv('output/'+classProperty+'_featureImportances.csv')
# print('Feature Importances: ', '\n', featureImportances)
plt = featureImportances[:10].plot(x='Covariates', y='Feature_Importance', kind='bar', legend=False,
                              title='Feature Importances')
fig = plt.get_figure()
fig.savefig('output/'+classProperty+'_FeatureImportances.png', bbox_inches='tight')

# Classify the image
classifiedImageSingleMap = compositeToClassify.classify(trainedClassiferForSingleMap,classProperty+'_Predicted')


# Load the best model from the classifier list
classifierToBootstrap = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName','equals',bestModelName).first()).get('c'))

def pd_to_fc(df):
    df['features'] = df.apply(lambda row: ee.Feature(None,row.to_dict()), 1)
    featureList = df["features"].tolist()

    fc =  ee.FeatureCollection(featureList)
    return fc

fcList_woUpload = []
for n in seedsToUseForBootstrapping:
    # Perform the subsetting
    stratSample = preppedCollection.groupby(stratificationVariableString, group_keys=False).apply(lambda x: x.sample(n=int(round((strataDict.get(x.name)/100)*bootstrapModelSize)), replace=True, random_state=n))

    fc = pd_to_fc(stratSample)

    fcList_woUpload.append(fc)

def bootstrapFunc(fc):
    # Train the classifier with the collection
    trainedClassifer = classifierToBootstrap.train(fc,classProperty,covariateList)

    # Classify the image
    classifiedImage = compositeToClassify.classify(trainedClassifer,classProperty+'_Predicted')

    return classifiedImage

meanImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList_woUpload))).reduce(
    reducer = ee.Reducer.mean()
)

upperLowerCIImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList_woUpload))).reduce(
    reducer = ee.Reducer.percentile([2.5,97.5],['lower','upper'])
)

stdDevImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList_woUpload))).reduce(
    reducer = ee.Reducer.stdDev()
)

# PCA interpolation-extrapolation image
PCA_int_ext = assessExtrapolation(preppedCollection[covariateList], compositeToClassify, propOfVariance).rename('pct_int_ext')

# Construct final image
finalImageToExport = ee.Image.cat(classifiedImageSingleMap,
meanImage,
upperLowerCIImage,
stdDevImage,
PCA_int_ext)

exportTask = ee.batch.Export.image.toAsset(
    image = finalImageToExport.toFloat(),
    description = classProperty+'_Bootstrapped_MultibandImage',
    assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_Bootstrapped_MultibandImage' ,
    crs = 'EPSG:4326',
    crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
    region = exportingGeometry,
    maxPixels = int(1e13),
    pyramidingPolicy = {".default": pyramidingPolicy}
);
exportTask.start()

exportTask = ee.batch.Export.image.toDrive(
    image = finalImageToExport.multiply(100).toInt().toFloat().divide(100),
    description = classProperty+'_Bootstrapped_MultibandImage',
    crs = 'EPSG:4326',
    crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
    region = exportingGeometry,
    maxPixels = int(1e13)
);
exportTask.start()
