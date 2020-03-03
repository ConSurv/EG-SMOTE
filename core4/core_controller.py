import time
from core4 import gsom as GSOM_Core


class Controller:

    def __init__(self, params):
        self.params = params
        self.gsom_nodemap = None

    def _grow_gsom(self, inputs,X_test, dimensions, plot_for_itr=0, classes=None, output_loc=None):
        gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), inputs, dimensions, plot_for_itr=plot_for_itr, activity_classes=classes, output_loc=output_loc)
        gsom.grow()
        gsom.smooth()
        self.gsom_nodemap = gsom.assign_hits()
        gsom.finalize_gsom_label()
        y_pred = gsom.predict(X_test)
        return y_pred


    def run(self, input_vector_db, X_test,plot_for_itr=0, classes=None, output_loc=None):

        results = []
        y_pred = []

        for batch_key, batch_vector_weights in input_vector_db.items():

            batch_id = int(batch_key)

            start_time = time.time()
            y_pred = self._grow_gsom(batch_vector_weights,X_test, batch_vector_weights.shape[1], plot_for_itr=plot_for_itr, classes=classes, output_loc=output_loc)
            print('Batch', batch_id)
            print('Neurons:', len(self.gsom_nodemap))
            print('Duration:', round(time.time() - start_time, 2), '(s)\n')

            results.append({
                'gsom': self.gsom_nodemap,
                'aggregated': None
            })

        return results,y_pred




