def extend_lazy_auto_mapping(mapping, key, config_value, value):
    prev_config_mapping = mapping._config_mapping
    prev_model_mapping = mapping._model_mapping

    prev_config_mapping[key] = config_value
    prev_model_mapping[key] = value

def test_extending():
    import transformers
    extend_lazy_auto_mapping(transformers.models.auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, 'test', transformers.models.bert.configuration_bert, transformers.models.bert.BertForSequenceClassification)

if __name__ == "__main__":
    test_extending()