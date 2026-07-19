import json

d = json.load(open('modelos/relatorios/rfe_feature_ranking.json', encoding='utf-8'))
print('N features optimal:', d['n_features_optimal'])
print('Best CV accuracy:', d['best_cv_accuracy'])
print('N features total:', d['n_features_total'])
print()
print('Top 25 features por ranking:')
for f in d['features_ranking'][:25]:
    sel = 'SEL' if f['selected'] else '   '
    print(f"  [{sel}] rank={f['ranking']:3d}  {f['feature']}")
