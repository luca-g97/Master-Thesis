import ipywidgets as widgets
from IPython.display import display, clear_output

trainDataSet, testDataSet, maxTrain, maxTest, batchSizeTraining, batchSizeTest, trainingsLink, testLink, datasets = [[""]], [[""]], "", "", "", "", "", "", [[""]]
tab_nest, datasetTab, networkTab, trainingTab, visualizationTab = "", "", "", "", ""

datasetMapping = {
    "MNIST": {
        "train_samples": 10000,
        "test_samples": 2000,
        "epochs": 10,
        "learning_rate": 0.001,
        "eval_samples": 1,
        "closest_sources": 10,
        "show_closest_sources": 3
    }, "HSV-RGB": {
        "train_samples": 10000,
        "test_samples": 2000,
        "epochs": 10,
        "learning_rate": 0.001,
        "eval_samples": 1,
        "closest_sources": 10,
        "show_closest_sources": 3
    }, "Small 1x1": {
        "train_samples": 80,
        "test_samples": 20,
        "epochs": 10,
        "learning_rate": 0.001,
        "eval_samples": 1,
        "closest_sources": 10,
        "show_closest_sources": 3
    }, "WikiText2 (GPT2)": {
        "train_samples": 2,
        "test_samples": 5,
        "epochs": 10,
        "learning_rate": 0.0004,
        "eval_samples": 1,
        "closest_sources": 2,
        "show_closest_sources": 2
    }, "WikiText2 (LSTM)": {
        "train_samples": 100,
        "test_samples": 2000,
        "epochs": 100,
        "learning_rate": 0.001,
        "eval_samples": 1,
        "closest_sources": 10,
        "show_closest_sources": 3
    }
}

def createIntSlider(value, max, description, step=1, min=1, disabled=False):
    slider = widgets.IntSlider(
        value=value,
        min=min,
        max=max,
        step=step,
        description=description,
        disabled=disabled,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        continuous_update=False,
        layout=widgets.Layout(width='95%'),
        style = {'description_width': 'initial'}
    )
    return slider

def createSelectionSlider(value, options, description):
    slider = widgets.SelectionSlider(
        options=options,
        value=value,
        description=description,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        layout=widgets.Layout(width='95%'),
        style = {'description_width': 'initial'}
    )
    return slider

def createLayerChoice(options, tooltips=[], description="Type"):
    layerChoice = widgets.ToggleButtons(
        options=options,
        description=description,
        disabled=False,
        tooltips=tooltips,
        style = {'description_width': 'initial'}
    )
    return layerChoice

def createRangeSliderChoice(max, description):
    options = [f"{i}" for i in range(max)]
    slider = widgets.SelectionRangeSlider(
        options=options,
        index=(0, (max-1)),
        description=description,
        disabled=False,
        layout=widgets.Layout(width='70%'),
        style = {'description_width': 'initial'}
    )
    return slider


def createBoolButtonChoice(description, tooltip, value = False, disabled=False, icon='check'):
    boolButton = widgets.ToggleButton(
        value=value,
        description=description,
        disabled=disabled,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip=tooltip,
        #layout=widgets.Layout(width='20%'),
        style = {'description_width': 'initial'},
        layout=widgets.Layout(width='20%', margin='0% 0% 0% 5%')
        #icon='check' # (FontAwesome names without the `fa-` prefix)
    )
    return boolButton

datasetChoice = createLayerChoice(options=['MNIST', 'HSV-RGB', 'WikiText2 (GPT2)', 'WikiText2 (LSTM)'], #'Small1x1' 
                                  tooltips=['60000 Written numbers from 0 to 9 (Classification)', '50000 HSV-values (Conversion to RGB)',
                                            'Wikipedia (WikiText2) Dataset using GPT2-Architecture', 'Wikipedia (WikiText2) Dataset using LSTM-Architecture'], description='Dataset') #'90 random values of the small 1x1 (Calculation)'

def updateDatasetTab():
    global datasetChoice

    return widgets.VBox([datasetChoice])

num_layers = 2
layerAmountChoice = createIntSlider(num_layers, min=1, max=16, description="Layer")
seeds = ["Random"] + [i for i in range(100)]
seedChoice = createSelectionSlider(value="Random", options=seeds, description="Seed")
useBitLinearChoice = createBoolButtonChoice(description="Activate BitLinear", tooltip="Use BitLinear instead of nn.Linear")
normalLayerSizeChoice = [createIntSlider(value=128, min=1, max=1024, description="Size", step=1)
                         for _ in range(num_layers)]
normalLayerChoice = [createLayerChoice(options=['Linear'],
                                       tooltips=['Fully connected layer'], description='Type') for _ in range(num_layers)]
outputLayerChoice = createLayerChoice(options=['Linear'], tooltips=['Fully connected layer'], description='Type')
activationLayerChoice = [createLayerChoice(options=['ReLU', 'Sigmoid', 'Tanh', 'None'],
                                           tooltips=['only Values >= 0', 'only Values between 0 and 1', 'normalized Values (between -1 and 1)'],
                                           description='Type') for _ in range(num_layers)]
outputActivationLayerChoice = createLayerChoice(options=['ReLU', 'Sigmoid', 'Tanh', 'None'],
                                                tooltips=['only Values >= 0', 'only Values between 0 and 1', 'normalized Values (between -1 and 1)'], description='Type')

# Function to update tab children based on the number of layers
def updateNetworkTab():
    global normalLayerChoice
    global normalLayerSizeChoice
    global activationLayerChoice
    global datasetChoice

    num_layers = layerAmountChoice.value

    # Create list of ToggleButtons for normal and activation choices for each layer
    normalLayerChoice = [createLayerChoice(options=['Linear'],
                                           tooltips=['Fully connected layer'], description='Type') for _ in range(num_layers)]

    normalLayerSizeChoice = [createIntSlider(value=128, min=1, max=1024, description="Size", step=1)
                             for _ in range(num_layers)]

    conv2dLayerInputSizeChoice = [createIntSlider(value=128, min=1, max=1024, description="Input Size", step=1)
                                  for _ in range(num_layers)]
    conv2dLayerOutputSizeChoice = [createIntSlider(value=128, min=1, max=1024, description="Output Size", step=1)
                                   for _ in range(num_layers)]
    kernelSizeChoice = [createIntSlider(value=3, min=1, max=7, description="Kernel Size", step=1)
                        for _ in range(num_layers)]

    activationLayerChoice = [createLayerChoice(options=['ReLU', 'Sigmoid', 'Tanh', 'None'],
                                               tooltips=['only Values >= 0', 'only Values between 0 and 1', 'normalized Values (between -1 and 1)'],
                                               description='Type') for _ in range(num_layers)]

    # Create a VBox for each layer with normal and activation choices
    normalLayerVBox = [widgets.VBox([normalLayerChoice[i], normalLayerSizeChoice[i]]) for i in range(num_layers)]
    activationLayerVBox = [widgets.VBox([activationLayerChoice[i]]) for i in range(num_layers)]
    outputLayerVBox = widgets.VBox([outputLayerChoice])
    outputActivationLayerVBox = widgets.VBox([outputActivationLayerChoice])

    # Create list of Accordions for normal and activation layers
    accordionNormalLayer = [widgets.Accordion(children=[normalLayerVBox[i]]) for i in range(num_layers)]
    accordionActivationLayer = [widgets.Accordion(children=[activationLayerVBox[i]]) for i in range(num_layers)]
    accordionOutputLayer = widgets.Accordion(children=[outputLayerVBox])
    accordionOutputActivationLayer = widgets.Accordion(children=[outputActivationLayerVBox])

    tab_nest = widgets.Tab()
    tab_nest.children = [widgets.VBox([accordionNormalLayer[i], accordionActivationLayer[i]]) for i in range(num_layers)] + [widgets.VBox([accordionOutputLayer, accordionOutputActivationLayer])]

    # Set titles for the Accordions
    for i in range(num_layers+1):
        if(i == num_layers):
            tab_nest.set_title(i, f"Output")
            accordionOutputLayer.set_title(0, f'Output Layer')
            accordionOutputActivationLayer.set_title(0, f'Activation Layer')
        else:
            accordionNormalLayer[i].set_title(0, f'Normal Layer')
            accordionActivationLayer[i].set_title(0, f'Activation Layer')
            #chosenLayers.append((accordionNormalLayer[i].children[0].children[1].value, accordionActivationLayer[i].children[0].children[0].value))
            tab_nest.set_title(i, f"Layer {i}")

    if(datasetChoice.value == "WikiText2 (GPT2)" or datasetChoice.value == "WikiText2 (LSTM)"):
        return widgets.VBox([seedChoice, useBitLinearChoice, layerAmountChoice])
    else:
        return widgets.VBox([seedChoice, useBitLinearChoice, layerAmountChoice, tab_nest])

trainSamplesChoice = createIntSlider(2000, min=1, max=10, description="Train Samples")
testSamplesChoice = createIntSlider(2000, min=1, max=10, description="Test Samples")

def initializeTrainSet(trainSetMNIST, testSetMNIST):
    global datasetChoice, trainDataSet, testDataSet, maxTrain, maxTest, trainSamplesChoice, testSamplesChoice, batchSizeTraining, batchSizeTest, trainingsLink, testLink
    trainDataSet = trainSetMNIST
    testDataSet = testSetMNIST
    maxTrain = len(trainDataSet)
    maxTest = len(testDataSet)
    trainSamplesChoice = createIntSlider(10000, min=1, max=maxTrain, description="Train Samples")
    testSamplesChoice = createIntSlider(2000, min=1, max=maxTest, description="Test Samples")
    batchSizeTraining = createIntSlider(64, min=1, max=maxTrain, description="Batchsize Training")
    batchSizeTest = createIntSlider(64, min=1, max=maxTest, description="Batchsize Test")

epochsChoice = createIntSlider(10, min=1, max=1000, description="Epochs")
learningRateChoice = widgets.BoundedFloatText(value=0.001, min=0.0000001, max=10.0, step=0.0000001, description='Learning Rate', style = {'description_width': 'initial'}, disabled=False)

lossChoice = createLayerChoice(options=['MSE', 'Cross-Entropy'],
                               tooltips=['Average Squared difference between the actual and predicted values', 'Difference between the actual and predicted probability distributions'], description='Loss-Type')
optimizerChoice = createLayerChoice(options=['Adam', 'SGD'],
                                    tooltips=['Adaptive learning rate and efficient handling of sparse gradients', 'SGD is best when the dataset is large, and computation efficiency is crucial'], description='Optimizer-Type')

def updateTrainingTab():
    global datasetChoice, trainSamplesChoice, testSamplesChoice, trainDataSet, testDataSet, batchSizeTraining, batchSizeTest, learningRateChoice, optimizerChoice

    maxTrain = len(trainDataSet)
    maxTest = len(testDataSet)

    learningRateChoice = widgets.BoundedFloatText(value=datasetMapping[datasetChoice.value]["learning_rate"], min=0.0000001, max=10.0, step=0.0000001, description='Learning Rate', style = {'description_width': 'initial'}, disabled=False)
    trainSamplesChoice = createIntSlider(datasetMapping[datasetChoice.value]["train_samples"], min=1, max=maxTrain, description="Train Samples")
    testSamplesChoice = createIntSlider(datasetMapping[datasetChoice.value]["test_samples"], min=1, max=maxTest, description="Test Samples")
    batchSizeTraining = createIntSlider(64, min=1, max=maxTrain, description="Batchsize Training")
    batchSizeTest = createIntSlider(64, min=1, max=maxTest, description="Batchsize Test")
    trainingsLink = widgets.jslink((trainSamplesChoice, 'value'), (batchSizeTraining, 'max'))
    testLink = widgets.jslink((testSamplesChoice, 'value'), (batchSizeTest, 'max'))

    if(datasetChoice.value == "WikiText2 (GPT2)" or datasetChoice.value == "WikiText2 (LSTM)"):
        return widgets.VBox([epochsChoice, learningRateChoice, trainSamplesChoice, testSamplesChoice])
    else:
        return widgets.VBox([epochsChoice, learningRateChoice, trainSamplesChoice, testSamplesChoice, batchSizeTraining,
                             batchSizeTest, lossChoice, optimizerChoice])

visualizationChoice = createLayerChoice(options=['Weighted', 'Per Neuron & Layer', 'Per Neuron Only', 'Per Layer Only', 'Custom'],
                                        tooltips=['Used ideally for ...', 'Used also ideally for ...', 'Used also ideally for ...'], description='Visualize')
neuronChoice = [createRangeSliderChoice(max=1024, description=f"Layer {i}") for i in range(num_layers)]
activationLayerTypeChoice = [createBoolButtonChoice(description="ReLU", tooltip="Include activation layer") for i in range(num_layers)]
outputLayerSizeChoice=createRangeSliderChoice(max=10, description=f"Output Layer")
outputLayerActivationChoiceType=createBoolButtonChoice(description=f"Activation layer (ReLU)", tooltip="Include activation layer", disabled = True if outputActivationLayerChoice.value == "None" else False)
layerChoice = [widgets.HBox([neuronChoice[i], activationLayerTypeChoice[i]]) for i in range(num_layers)] + [widgets.HBox([outputLayerSizeChoice, outputLayerActivationChoiceType])]
customBox = widgets.VBox(children=layerChoice)
evalSamplesChoice = createIntSlider(1, min=1, max=100, description="Evaluations")
closestSourceTestLink = widgets.jslink((testSamplesChoice, 'value'), (evalSamplesChoice, 'max'))
closestSourcesChoice = createIntSlider(42, min=1, max=10000, description="Closest Sources")
closestSourceTrainingLink = widgets.jslink((trainSamplesChoice, 'value'), (closestSourcesChoice, 'max'))
showClosestMostUsedSourcesChoice = createIntSlider(3, min=1, max=20, description="Show Closest Sources")
showClosestMostUsedSourcesLink = widgets.jslink((trainSamplesChoice, 'value'), (showClosestMostUsedSourcesChoice, 'max'))

def updateVisualizationTab():
    global normalLayerChoice, normalLayerSizeChoice, activationLayerChoice, neuronChoice, activationLayerTypeChoice, outputLayerSizeChoice, outputLayerActivationChoiceType, closestSourcesChoice, trainSamplesChoice, testSamplesChoice, evalSamplesChoice, showClosestMostUsedSourcesChoice, closestSourceTrainingLink

    evalSamplesChoice = createIntSlider(datasetMapping[datasetChoice.value]["eval_samples"], min=1, max=len(testDataSet), description="Evaluations")
    outputLayerSizeChoice = createRangeSliderChoice(max=10, description=f"Output Layer")
    outputLayerActivationChoiceType = createBoolButtonChoice(description=f"Activation layer ({outputActivationLayerChoice.value})", tooltip="Include activation layer", disabled = True if outputActivationLayerChoice.value == "None" else False)

    evalSamplesChoice = createIntSlider(datasetMapping[datasetChoice.value]["eval_samples"], min=1, max=100, description="Evaluations")
    closestSourceTestLink = widgets.jslink((testSamplesChoice, 'value'), (evalSamplesChoice, 'max'))
    closestSourcesChoice = createIntSlider(datasetMapping[datasetChoice.value]["closest_sources"], min=1, max=10000, description="Closest Sources")
    closestSourceTrainingLink = widgets.jslink((trainSamplesChoice, 'value'), (closestSourcesChoice, 'max'))
    showClosestMostUsedSourcesChoice = createIntSlider(datasetMapping[datasetChoice.value]["show_closest_sources"], min=1, max=20, description="Show Closest Sources")
    showClosestMostUsedSourcesLink = widgets.jslink((trainSamplesChoice, 'value'), (showClosestMostUsedSourcesChoice, 'max'))

    if(datasetChoice.value == "WikiText2 (GPT2)" or datasetChoice.value == "WikiText2 (LSTM)"):
        return widgets.VBox([evalSamplesChoice, closestSourcesChoice, showClosestMostUsedSourcesChoice])

    if(visualizationChoice.value == "Custom"):
        num_layers = layerAmountChoice.value
        neuronChoice = [createRangeSliderChoice(max=normalLayerSizeChoice[i].value, description=f"Neurons in Layer {i} ({normalLayerChoice[i].value})")
                        for i in range(num_layers)]
        activationLayerTypeChoice = [createBoolButtonChoice(description=f"Activation layer ({activationLayerChoice[i].value})",
                                                            tooltip="Click to include activation layer in the output", disabled = True if activationLayerChoice[i].value == "None" else False)
                                     for i in range(num_layers)]
        layerChoice = [widgets.HBox([neuronChoice[i], activationLayerTypeChoice[i]]) for i in range(num_layers)] + [widgets.HBox([outputLayerSizeChoice, outputLayerActivationChoiceType])]
        customBox = widgets.VBox(children=layerChoice)
        return widgets.VBox([evalSamplesChoice, closestSourcesChoice, showClosestMostUsedSourcesChoice, visualizationChoice, customBox])
    else:
        return widgets.VBox([evalSamplesChoice, closestSourcesChoice, showClosestMostUsedSourcesChoice, visualizationChoice])

def changeTab(change):
    if change['name'] == 'selected_index' and change['new'] == 3:
        updateChoiceTabs(3)

def observeTabChange():
    if hasattr(observeTabChange, 'tab_observer'):
        tab_nest.unobserve(observeTabChange.tab_observer)
    observeTabChange.tab_observer = tab_nest.observe(changeTab, names='selected_index')

def updateChoiceTabs(index, jumpToIndex = True):
    global tab_nest
    global datasetTab
    global networkTab
    global trainingTab
    global visualizationTab

    clear_output()
    tab_nest = widgets.Tab()

    if(index == 0):
        datasetTab = updateDatasetTab()
    elif(index == 1):
        networkTab = updateNetworkTab()
        visualizationTab = updateVisualizationTab()
    elif(index == 2):
        trainingTab = updateTrainingTab()
    elif(index == 3):
        visualizationTab = updateVisualizationTab()

    tab_nest.children = [datasetTab, networkTab, trainingTab, visualizationTab]

    tabNames = ("Datasamples", "Network", "Training", "Visualization")
    for i, name in enumerate(tabNames):
        tab_nest.set_title(i, name)

    if jumpToIndex == True:
        tab_nest.selected_index = index
    tab_nest.unobserve(observeTabChange.tab_observer)
    tab_nest.observe(changeTab, names='selected_index')

    display(tab_nest)

def changeNetworkTab(change):
    updateChoiceTabs(1)

def changeTrainingTab(change):
    global trainDataSet
    global testDataSet

    trainDataSet, testDataSet = datasets[datasetChoice.value]
    updateChoiceTabs(2, jumpToIndex=False)
    updateChoiceTabs(1, jumpToIndex=False)
    updateChoiceTabs(3, jumpToIndex=False)

def changeVisualizationTab(change):
    updateChoiceTabs(3)

def initialize(trainSetMNIST, testSetMNIST, datasetsParameter):
    global tab_nest, datasetTab, networkTab, trainingTab, visualizationTab, datasets
    datasets = datasetsParameter
    initializeTrainSet(trainSetMNIST, testSetMNIST)
    # @title Click `Show code` in the code cell. { display-mode: "form" }
    tab_nest = widgets.Tab()
    datasetTab = updateDatasetTab()
    networkTab = updateNetworkTab()
    trainingTab = updateTrainingTab()
    visualizationTab = updateVisualizationTab()
    observeTabChange()
    # Display the initial widgets
    updateChoiceTabs(0)

    # Observe all necessary values
    layerAmountChoice.observe(changeNetworkTab, names='value')
    visualizationChoice.observe(changeVisualizationTab, names='value')
    datasetChoice.observe(changeTrainingTab, names='value')



