import yaml
def conf_load(filename):
	with open(filename, 'r') as stream:
		try:
			ga_params = yaml.safe_load(stream)

			#getting generation zero from run zero
			generation_zero = ga_params['runs'][0]['generations'][0]
			# print(generation_zero)

			#getting rssnp index 0 in generation zero
			rssnp_zero = generation_zero['rssnp_chromosomes'][0]
			# print(rssnp_zero) 

			#getting out_pairs indexed 0 at rssnp indexed 0 in generation zero
			out_pairs_zero = rssnp_zero['out_pairs'][0]
			# print(out_pairs_zero)
			return generation_zero, rssnp_zero, out_pairs_zero
		except yaml.YAMLError as exc:
			print(exc)

filename = 'ga_conf_out.yaml'
generation ,rssnp , out_pairs = conf_load(filename)

print(generation)
print('\n')
print(rssnp)
print(out_pairs)

for a in rssnp:
	print(a)

for a in out_pairs:
	print(a)

