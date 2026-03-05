from .gradient_boosting import (
    train_xgboost,
    train_xgboost_cv,
    train_lightgbm,
    train_lightgbm_cv,
    evaluate_boosting_model,
    get_feature_importance,
    compute_shap_values,
)

from .hyperparameter_tuning import (
    optimize_xgboost,
    optimize_lightgbm,
    get_feature_importance as ht_get_feature_importance,
    compute_shap_values as ht_compute_shap_values,
)

from .ensemble import (
    load_base_models,
    generate_base_predictions,
    train_meta_learner,
    EnsemblePredictor,
    StackingEnsemble,
    BlendingEnsemble,
)

from .evaluation import (
    run_backtest,
    compute_position_wise_mae,
    analyze_price_brackets,
    calculate_expected_rank_improvement,
    compare_models,
    compute_cross_validation_results,
    analyze_feature_effects,
    analyze_error_distribution,
    compute_prediction_intervals,
    generate_evaluation_report,
)

# Optional import: NN stack should not block non-NN workflows.
try:
    from .lstm_model import (
        FPLLSTM,
        FPLLSTMWithAttention,
        train_lstm,
        predict_lstm,
        save_lstm_model,
        load_lstm_model,
    )
except ImportError:
    pass
