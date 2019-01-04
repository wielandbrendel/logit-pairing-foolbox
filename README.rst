=======
Adversarial Logit Pairing for Foolbox
=======
This repository allows it to load the `official pretrained ResNet-50 v2 with Adversarial Logit Pairing <https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing>`__ as a Foolbox model using see Foolbox model API.

.. code-block:: python

   from foolbox import zoo
   url = "https://github.com/wielandbrendel/logit-pairing-foolbox.git"
   model = zoo.get_model(url)

Images should be fed with size 64 x 64 x 3 and with pixel values in the range [0, 255].
