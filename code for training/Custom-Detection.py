from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()    ##create an instance of the CustomImagePrediction class

prediction.setModelTypeAsResNet()       
##sets the model type of the image recognition instance you created to the ResNet model, 
##which means you will be performing your image prediction tasks using the “ResNet” model 
##generated during your custom training

prediction.setModelPath(os.path.join(execution_path, "model_ex-001_acc-1.000000.h5"))
##accepts a string which must be the path to the model file generated during
##your custom training and must corresponds to the model type you set for your image prediction instance

prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
##accepts a string which must be the path to the JSON file generated during your custom training

prediction.loadModel(num_objects=2)
##loads the model from the path you specified in the function call above into your
##image prediction instance. You will have to set the parameter num_objects to the number of classes in your image datase

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "7.jpg"), result_count=5)
##performs actual prediction of an image. It can be called many times 
##on many images once the model as been loaded into your prediction instance

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
    
