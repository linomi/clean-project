from tensorflow.keras.callbacks import callback
import numpy as np
class my_callback(callback):
    def __init__(self,core_model,core_model_path,validation_data,eval_history_path):
        super().__init__()
        self.core_model = core_model
        self.core_model_path = core_model_path
        self.validation_data = validation_data
        self.eval_history_path = eval_history_path
        self.eval_history = np.load(eval_history_path)
    def on_train_end(self,**kwargs)
        super().on_train_end(self,**kwargs)
        result = self.model.evaluate(self.validation_data)
        if result <= self.eval_history.min():
            self.core_model.save(self.core_model_path,overwrtie = True)
            self.eval_history = np.append(self.eval_history,result)
            np.save(self.eval_history_path,self.eval_history)
