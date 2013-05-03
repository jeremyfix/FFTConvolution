
PLOT_SCRIPTS=\
	plot_comparisons.py
#	plot_fftw_convolution.py\
#	plot_gsl_convolution.py \
#	plot_std_convolution.py

plots:$(PLOT_SCRIPTS)
	@for plot_script in $(PLOT_SCRIPTS); do \
		python $$plot_script; \
	done
