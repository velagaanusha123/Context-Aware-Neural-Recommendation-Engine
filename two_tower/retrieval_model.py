import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_recommenders as tfrs

# =====================================================
# USER TOWER
# =====================================================

class UserModel(tf.keras.Model):

    def __init__(self, max_user_id, max_age):
        super().__init__()

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=max_user_id + 2,
            output_dim=128
        )

        self.age_embedding = tf.keras.layers.Embedding(
            input_dim=max_age + 2,
            output_dim=16
        )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64)
        ])

    def call(self, inputs):

        user_vec = self.user_embedding(
            tf.cast(inputs["user_idx"], tf.int32)
        )

        age_vec = self.age_embedding(
            tf.cast(inputs["age"], tf.int32)
        )

        x = tf.concat([user_vec, age_vec], axis=-1)

        return self.dense(x)

# =====================================================
# ITEM TOWER
# =====================================================

class ItemModel(tf.keras.Model):

    def __init__(self,
                 max_item_id,
                 max_product_group,
                 max_colour_group,
                 max_garment_group):
        super().__init__()

        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=max_item_id + 2,
            output_dim=128
        )

        self.product_embedding = tf.keras.layers.Embedding(
            input_dim=max_product_group + 2,
            output_dim=16
        )

        self.colour_embedding = tf.keras.layers.Embedding(
            input_dim=max_colour_group + 2,
            output_dim=16
        )

        self.garment_embedding = tf.keras.layers.Embedding(
            input_dim=max_garment_group + 2,
            output_dim=16
        )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64)
        ])

    def call(self, inputs):

     item_vec = self.item_embedding(
        tf.cast(tf.reshape(inputs["item_idx"], [-1]), tf.int32)
    )

     product_vec = self.product_embedding(
        tf.cast(tf.reshape(inputs["product_group_name"], [-1]), tf.int32)
    )

     colour_vec = self.colour_embedding(
        tf.cast(tf.reshape(inputs["colour_group_name"], [-1]), tf.int32)
    )

     garment_vec = self.garment_embedding(
        tf.cast(tf.reshape(inputs["garment_group_name"], [-1]), tf.int32)
    )

     x = tf.concat(
    [
        item_vec,
        product_vec,
        colour_vec,
        garment_vec
    ],
    axis=-1
)

     return self.dense(x)

# =====================================================
# RETRIEVAL MODEL (FIXED)
# =====================================================

class FashionRetrievalModel(tfrs.models.Model):

    def __init__(self, user_model, item_model, candidate_dataset):
        super().__init__()

        self.user_model = user_model
        self.item_model = item_model

        self.task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(
        candidates=candidate_dataset.batch(
            256,
            drop_remainder=True
        ).map(self.item_model)
    )
)

    def compute_loss(self, features, training=False):

        # FIX: correct feature separation
        user_inputs = {
            "user_idx": features["user_idx"],
            "age": features["age"]
        }

        item_inputs = {
            "item_idx": features["item_idx"],
            "product_group_name": features["product_group_name"],
            "colour_group_name": features["colour_group_name"],
            "garment_group_name": features["garment_group_name"]
        }

        user_embeddings = self.user_model(user_inputs)
        item_embeddings = self.item_model(item_inputs)

        return self.task(user_embeddings, item_embeddings)