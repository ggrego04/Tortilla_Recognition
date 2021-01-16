from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()         ##create an instance of the ModelTraining class

model_trainer.setModelTypeAsResNet()    ##set your instance property and start the traning process. 
                                        ##this function sets the model type of the training instance you created to the ResNet
                                        ##model, which means the ResNet algorithm will be trained on your dataset
                                        
model_trainer.setDataDirectory(r"C:\Users\Andreas Thoma\Desktop\Mathimata\EPL445\Project\Tortillas")  
## accepts a string which must be the path to the folder that contains the test and train subfolder of your image dataset


model_trainer.trainModel(num_objects=2, num_experiments=5, enhance_data=True, batch_size=32, show_network_summary=True)
##this is the function that starts the training process. Once it starts, it will create a JSON file in
##the dataset/json folder (e.g Tortillas/json) which contains the mapping of the classes of the dataset. 
##The JSON file will be used during custom prediction to produce reults

