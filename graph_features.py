

import pandas as pd
import numpy as np
import networkx as nx

WINDOW_HOURS = 48


def build_ring_features(df, window_hours=WINDOW_HOURS, min_shared_types=2):
    """
    Connects two accounts only if they share at least `min_shared_types`
    DIFFERENT entity types (e.g. same device AND same IP) within the window.
    A single shared resource (common device, common payment gateway) is not
    enough -- that happens constantly in normal traffic because the resource
    pools are small. Two or more independent overlaps together is what your
    synthetic fraud clusters were actually built to do, and it's a much
    rarer coincidence for legitimate customers.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_bin"] = df["timestamp"].dt.floor(f"{window_hours}h")

    ring_size = np.zeros(len(df), dtype=int)

    for _bin, group in df.groupby("_bin"):
        acct_entities = {}
        for row in group.itertuples():
            acct = row.account_id
            ents = acct_entities.setdefault(acct, set())
            ents.add(("dev", row.device_id))
            ents.add(("ip", row.ip_id))
            ents.add(("pmt", row.payment_method_id))
            if row.promotion_id != -1:
                ents.add(("promo", row.promotion_id))

        entity_to_accts = {}
        for acct, ents in acct_entities.items():
            for e in ents:
                entity_to_accts.setdefault(e, set()).add(acct)

        pair_shared_types = {}
        for e, accts in entity_to_accts.items():
            if len(accts) < 2 or len(accts) > 50:
                # skip entities touched by too many accounts in-window (uninformative
                # pool collision) -- keeps this tractable and meaningful
                continue
            accts = list(accts)
            etype = e[0]
            for i in range(len(accts)):
                for j in range(i + 1, len(accts)):
                    key = tuple(sorted((accts[i], accts[j])))
                    pair_shared_types.setdefault(key, set()).add(etype)

        G = nx.Graph()
        G.add_nodes_from(acct_entities.keys())
        for (a, b), types_seen in pair_shared_types.items():
            if len(types_seen) >= min_shared_types:
                G.add_edge(a, b)

        acct_to_ring = {}
        for comp in nx.connected_components(G):
            for a in comp:
                acct_to_ring[a] = len(comp)

        for idx, row in zip(group.index, group.itertuples()):
            ring_size[idx] = acct_to_ring.get(row.account_id, 1)

    df["ring_size_48h"] = ring_size
    df = df.drop(columns="_bin")
    return df


if __name__ == "__main__":
    df = pd.read_csv("features_with_split.csv")
    df = build_ring_features(df)

    print("ring_size_48h -- fraud vs normal:")
    print(f"  fraud median:  {df.loc[df.is_fraud==1, 'ring_size_48h'].median()}")
    print(f"  normal median: {df.loc[df.is_fraud==0, 'ring_size_48h'].median()}")
    print(f"  fraud mean:    {df.loc[df.is_fraud==1, 'ring_size_48h'].mean():.2f}")
    print(f"  normal mean:   {df.loc[df.is_fraud==0, 'ring_size_48h'].mean():.2f}")

    df.to_csv("features_with_graph.csv", index=False)
    print("\nSaved features_with_graph.csv")
