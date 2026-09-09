import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def generate_campaign(
    T=2000,
    quiet_frac=(0.35, 0.55),
    noise_scale=0.50,
    signal_inc=0.02,
    front_frac=0.15,
    p_front=0.0015,
    p_quiet=0.0003,
    p_back=0.012,
    p_mid=0.0022,
    seed=0,
):
    rng = np.random.default_rng(seed)

    stage = 0
    stages = []
    for t in range(T):
        stages.append(stage)

        frac = t / T
        if stage >= 4:
            p_advance = 0.0
        elif frac < front_frac:
            p_advance = p_front
        elif quiet_frac[0] <= frac < quiet_frac[1]:
            p_advance = p_quiet
        elif frac >= quiet_frac[1]:
            p_advance = p_back
        else:
            p_advance = p_mid

        if rng.random() < p_advance:
            stage += 1

    stages = np.array(stages)

    n = T
    signal = 0.5 + signal_inc * stages + rng.normal(0, noise_scale, size=n)
    breadth = rng.poisson(lam=0.25 + 0.02 * stages)
    breadth = breadth.astype(float)
    reversible = (rng.random(n) < 0.85).astype(float)

    features = np.column_stack([signal, breadth, reversible])
    risk = (stages >= 3).astype(int)

    return {
        "stages": stages,
        "features": features,
        "risk": risk,
        "T": T,
    }


def generate_campaigns(num, T=2000, noise_scale=0.50, signal_inc=0.02,
                       base_seed=0, **kwargs):
    campaigns = []
    for i in range(num):
        c = generate_campaign(
            T=T,
            noise_scale=noise_scale,
            signal_inc=signal_inc,
            seed=base_seed + i,
            **kwargs,
        )
        campaigns.append(c)
    return campaigns


def apply_state(features, tau):
    T, d = features.shape
    alpha = np.exp(-1.0 / tau)
    state = np.zeros((T, d))
    for t in range(1, T):
        state[t] = alpha * state[t - 1] + (1.0 - alpha) * features[t - 1]
    return state


def main():
    T = 2000
    noise_scale = 0.50
    signal_inc = 0.02
    quiet_frac = (0.35, 0.55)
    front_frac = 0.15
    p_front = 0.0015
    p_quiet = 0.0003
    p_back = 0.012
    p_mid = 0.0022
    tau_slow = 2500

    n_train = 200

    print("Generating long training campaigns...")
    train = generate_campaigns(
        n_train,
        T=T,
        noise_scale=noise_scale,
        signal_inc=signal_inc,
        quiet_frac=quiet_frac,
        front_frac=front_frac,
        p_front=p_front,
        p_quiet=p_quiet,
        p_back=p_back,
        p_mid=p_mid,
        base_seed=3333,
    )

    y_train = np.concatenate([c["risk"] for c in train])

    print("Computing slow EMA state...")
    states_train = [apply_state(c["features"], tau_slow) for c in train]

    X_train_slow = np.vstack([
        np.hstack([c["features"], states_train[i]])
        for i, c in enumerate(train)
    ])

    print("Training slow stateful logistic model...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    model.fit(X_train_slow, y_train)

    print("Saving model to slow_ema_model.joblib...")
    joblib.dump(model, "models/slow_ema_model.joblib")
    print("Done.")


if __name__ == "__main__":
    main()