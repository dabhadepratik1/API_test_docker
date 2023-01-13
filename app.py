from flask import Flask,jsonify,request
from flask_restful import Api,Resource
from keras.models import load_model
import keras.utils as image


app=Flask(__name__)
api=Api(app)

model = load_model("mnist_ann.h5")

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(28,28),color_mode="grayscale")
	i = image.img_to_array(i)/255
	i = i.reshape(1, 28,28)
	p = model.predict(i).argmax(axis=1)[0]
	return p

class MakePrediction(Resource):
    @staticmethod
    def post():
        img=request.files['image']
        img_path = "static/" + img.filename
        img.save(img_path)
        prediction=predict_label(img_path)
        prediction=int(prediction)
        return jsonify({"Prediction ":prediction})

api.add_resource(MakePrediction, '/predict')

if __name__=='__main__':
	app.run(host="0.0.0.0",port=int("5000"),debug=True)
    