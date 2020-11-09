#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  test.py
#
#  Copyright 2020 - Loïc Vandenberghe - Mathieu Toubeix - Walid Ben Naceur
#
import model_full as m
import models_images_to_labels as m1
import models_labels_to_diff as m2
import tools
import trainer


def main(args):

    # ==== Arguments ====
    # (could not made it in command line because dlc_practical_prologue uses it)
    batch_size = 100
    epochs = 50
    n_data = 1000
    verbose = False
    plot = True
    repeat = 3
    aux_loss = 0.5
    eta = 1e-4

    # ==== End Arguments ====

    data = trainer.load_and_process_data(n_data)

    def train_show(Model1, Model2, aux=False, shared=True):
        best = None
        best_results = None

        train_errors = []
        test_errors = []

        for i in range(repeat):

            if shared:
                if aux:
                    model = m.CombineWithLabels(Model1(), Model2())
                else:
                    model = m.CombineNet(Model1(), Model2())
            else:
                if aux:
                    model = m.NotSharedCombineWithLabels(Model1(), Model1(), Model2())
                else:
                    model = m.NotSharedCombine(Model1(), Model1(), Model2())

            if aux:
                results = trainer.train_model_auxiliary_loss(model,
                                                             data["train_input"],
                                                             data["train_target_dif"].view(-1, 1),
                                                             data["train_target_label"],
                                                             data["test_input"], data["test_target_dif"],
                                                             epochs=epochs,
                                                             verbose=verbose,
                                                             mini_batch_size=batch_size,
                                                             aux_weight=0.2,
                                                             eta=eta)
            else:
                results = trainer.train_model(model,
                                              data["train_input"],
                                              data["train_target_dif"].view(-1, 1),
                                              data["test_input"], data["test_target_dif"],
                                              epochs=epochs,
                                              verbose=verbose,
                                              mini_batch_size=batch_size,
                                              eta=eta)

            train_errors += [results[1]]
            test_errors += [results[2]]

            min_test_error = results[2].min()
            end_test_error = results[2][-1]

            if repeat > 1:
                print("it = ", i)
            print("minimum test error=", min_test_error, " ( ", min_test_error * 100 / n_data, "%)")
            print("last test error (%)=", end_test_error, " ( ", end_test_error * 100 / n_data, "%)")
            print()

            if best is None or best > min_test_error:
                best = min_test_error
                best_results = results

        if plot:
            if repeat == 1:
                tools.plot(best_results[1], best_results[2])
            else:
                tools.plotAll(train_errors, test_errors)

    print("=======================================")
    print("       Fully Connected Network         ")
    print("          No shared weight             ")
    print("=======================================")
    train_show(m1.FullyConnected_I, m2.FullyConnected_II, shared=False)

    print("=======================================")
    print("       Convolutionnal  Network         ")
    print("           on labels only              ")
    print("=======================================")
    print("loss =", aux_loss, "*aux_loss + ", 1 - aux_loss, "*loss")
    trainer.test_model_size(m1.LeNet5(), m2.ArgMax())
    train_show(m1.LeNet5, m2.ArgMax, aux=True)

    print("=======================================")
    print("       Fully Connected Network         ")
    print("=======================================")
    train_show(m1.FullyConnected_I, m2.FullyConnected_II)

    print("=======================================")
    print("       Fully Connected Network         ")
    print("         with auxiliary loss           ")
    print("=======================================")
    print("loss =", aux_loss, "*aux_loss + ", 1 - aux_loss, "*loss")
    train_show(m1.FullyConnected_I, m2.FullyConnected_II, aux=True)

    print("=======================================")
    print("      Convolutionnal   Network         ")
    print("=======================================")
    train_show(m1.LeNet5, m2.FullyConnected_II)

    print("=======================================")
    print("       Convolutionnal  Network         ")
    print("         with auxiliary loss           ")
    print("=======================================")
    print("loss =", aux_loss, "*aux_loss + ", 1 - aux_loss, "*loss")
    train_show(m1.LeNet5, m2.FullyConnected_II, aux=True)


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
