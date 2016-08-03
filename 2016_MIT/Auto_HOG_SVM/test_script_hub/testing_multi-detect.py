
'Picking up Screw Driver'
'Putting down Screw Driver'
'The Hand is moving with out holding something'
'The hand is holding Scrw Driver doing something'

def model_dictionnary():
	['hand', 'ScrewDriver']

def multiDetector(img, multiModels):
	raw = img.copy()
	feature_object=[]
	for model in multiModels:
		box, position = model(img.copy)
		feature.append(box)

