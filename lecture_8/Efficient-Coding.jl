### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ fa084448-889d-11ee-3172-c3c4348be3b0
begin
	using PlutoUI, Distributions, DataFrames, Downloads, CSV, Compose, Base, Plots, LaTeXStrings, LogExpFunctions, Roots, Optim, LinearAlgebra, Printf

	TableOfContents()
end

# ╔═╡ 3c9e3ee3-3398-4792-ba27-ea0636627faa
md"""
# Van Hateren's regimes of efficient coding

In this notebook we illustrate some of the results in [Van Hateren, Biological Cybernetics 1992](https://www.doi.org/10.1007/BF00203134), and we show step-by-step how the data collected in  [Caramellino et al, eLife 2021](https://doi.org/10.7554/eLife.72081) can be nicely interpreted using this theory. This exercise is meant to bring together the Bayesian approach to modeling perception we developed in the first part of the course and the notions based on information theory (efficient coding) we developed in the second part.

Some of the code below is visible by default, but some is not. I have chosen to hide by default the code that I think is less illuminating about what's going on (inclusing most of the plotting and the data wrangling), and to leave visible by default the more conceptual parts that can be mapped more directly to the theory. However, remember that you can always download this notebook (by clicking the button on the top right) and open it locally using [Pluto.jl](https://plutojl.org/). This will allow you to see all the code, change it etc etc.

## Theory recap
Here I have implemented an interactive diagram to illustrate the ideas in [Van Hateren, Biological Cybernetics 1992](https://www.doi.org/10.1007/BF00203134), using the notation of [Hermundstad et al, eLife 2014](https://doi.org/10.7554/eLife.03722), with a small exception (see below).

The setting is that of the figure below:

$(Resource("https://raw.githubusercontent.com/epiasini/bmitns/main/lecture_8/Hermundstad2014_supplement_extract.png", :width => 600))

We have ``K`` channels, ``k\in\{1,2,\ldots,K\}``. You can think about them as ``K`` different neurons or neural pathways conveying information about different features of, say, a visual stimulus. For each channel, the input (we'll also call it "stimulus") is normally distributed around 0 with standard deviation ``s_k``.

```math
\text{stimulus (input) }\sim\mathcal{N}\left(0,s_k\right)
```

The input is corrupted by some "sampling noise", normally distributed with standard deviation ``n_s``. In the sensory periphery, this could be for instance some sensory transduction noise.

```math
\text{stimulus + sampling noise }\sim\mathcal{N}\left(0,\sqrt{s_k^2+n_s^2}\right)
```

Then, the signal gets processed by our neuron/neural system of interest. To keep things simple, we limit this neural system to a simple linear operation: the signal gets multiplied by a certain number, which we call **neural gain**. You can think of this as an amplification/attenuation operation on the signal. For convenience, we parameterize the gain by its **square** ``g_k``. In other words, the gain is ``\sqrt{g_k}``. Therefore, after encoding by our neuron/neural system we have:

```math
\text{encoded signal } \sim\mathcal{N}\left(0,\sqrt{g_k\left(s_k^2+n_s^2\right)}\right)
```

After encoding, the signal is transmitted downstream for further processing. For instance, if the previous steps describe the operation of sensory transduction and encoding in the retina, this step represents transmission down the optic nerve towards the rest of the brain. We assume that this step also introduces some degree of noise, normally distributed with standard deviation ``n_c``. Therefore, the final output of our system is

```math
\text{output } \sim\mathcal{N}\left(0,\sqrt{g_k\left(s_k^2+n_s^2\right)+n_c}\right)
```

Finally, we assume that there is some constraint to the sigal that we can transmit in our output. For example, there could be a metabolic cost to generating signals which limits the firing rates of our neurons. We model this cost by assuming that we have a certain **output budget**, expressed as a constraint on the total variance ``Q`` of the output:

```math
\begin{equation}
\tag{1}
Q = \sum_{k=1}^K\operatorname{Var}\left[\text{output}_k\right] = \sum_{k=1}^K \left[g_k\left(s_k^2+n_s^2\right)+n_c\right] \equiv\text{constant}
\end{equation}
```

The efficient coding problem for this setting is then: **what are the values of ``g_k`` that maximize the transmitted information ``I[\text{input}:\text{output}]`` if the output budget ``Q`` is fixed to a certain level?**

In class we have shown that under these assumptions the solution is

```math
\begin{equation}
\tag{2}
g_k = \frac{-(2+s_k^2)+\sqrt{s_k^4+4s_k^2/\Lambda}}{2(1+s_k^2)}
\end{equation}
```

if ``g_k`` thus defined is positive, otherwise ``g_k=0`` (i.e., we "kill" the channel). In this expression, ``\Lambda`` is a Lagrange multiplier that controls ``Q``, and we showed in class that necessarily ``0<\Lambda<1``.  To simplify the expression, we have also assumed that ``n_c=n_s=1`` (we say that the "noise floor", the minimum amount of output variance of a channel, is 1). In class also we showed that for a channel to have nonzero gain (``g_k>0``, that is, for that channel not to be "killed") the input signal strength must be above a critical threshold ``s_k > \sqrt{\Lambda/(1-\Lambda)}``.

What does it mean when we say that "``\Lambda`` is a Lagrange multiplier that controls ``Q``", in practice? Consider that ``Q`` always depends on the set of gain values ``\{g_k\}_i^K`` through the expression in Equation 1. But if ``g_k`` is given by Equation 2, where it is a function of ``\Lambda``, this means that ``Q`` depends on ``\Lambda`` through ``g_k``: we can write ``Q=Q(\{g_k(\Lambda)\})=Q(\Lambda)``. So for any value of ``\Lambda`` we have a solution ``g_k`` for each channel, and a corresponding value of the total output power ``Q``. This is an **implicit solution** of our problem (which, remember, was: *given a certain value of ``Q``, how should I pick ``g_k`` to maximise transmitted information?*). In practice, for a given (desired) value of the output power ``Q^*`` we can find ``g_k`` by changing the value of ``\Lambda`` until ``Q=Q(\Lambda)=Q^*`` (using Equation 1). We can call the value of ``\Lambda`` where this happens ``\Lambda^*``. The value of ``g_k`` is then given by ``g_k^*=g_k(\Lambda^*)``. You can see this process in action in the interactive demo below.

**Caveat on the notation:** compared to the notation in Hermundstad et al 2014, we parameterize the gain by its **squared** value, which we indicate with ``g_k``. By contrast, the expressions in the paper are written in terms of the absolute value of the gain ``|L_k|``. The correspondence between the two notations is simply ``g_k=|L_k|^2``.
"""

# ╔═╡ 7152e1e9-f83d-4600-8245-e09dca2efec4
begin
	"""
		gain(sₖ, Λ)

	Optimal gain √gₖ for a channel, as a function of the input power sₖ (standard deviation) and the Lagrange multiplier Λ.
	"""
	function gain(sₖ, Λ)
		if sₖ > √(Λ/(1-Λ))
			return √((-(2+sₖ^2) + √(sₖ^4 + 4sₖ^2/Λ)) / (2(1+sₖ^2)))
		else
			return 0
		end
	end

	"""
		output_power(sₖ, Λ)

	Output power for a single channel with optimal gain, if the input power is sₖ (standard deviation) and the Lagrange multiplier is Λ.
	"""
	function output_power(sₖ, Λ)
	    return gain(sₖ, Λ)^2 * (1+sₖ^2) + 1
	end

	"""
		total_output_power(s, Λ)

	Compute the total output power ``Q=Q(\\Lambda)`` for the van Hateren system defined by a set of input channels with power `s`.

	# Arguments
	- `s`: the stimulus power for all input channels, encoded as an array of standard deviations: s=``\\{s_1, \\ldots, s_K\\}``.
	- `Λ`: the Lagrange multiplier associated with ``Q``.
	"""
	function total_output_power(s, Λ)
	    return sum(output_power.(s, Λ))
	end
end

# ╔═╡ 6141a7c6-02a5-442b-807d-de05e47e6e85
md"""
### Interactive display of the solution as a function of ``\Lambda`` and {``s_k``}
In the plot below, we can look at the solution of the problem just described for a particular case where there are ``K=15`` channels. Each channel has a different input signal strength. However, all signal strengths can be scaled together with one parameter that we pass to the interactive plot. For instance, when this parameter is 1, the signal strengths of the 15 channels are regularly distributed in the interval ``[0.5, 2]``. More generally, for a certain value ``s`` of the signal strength parameter, they will be distributed in the interval ``[s/2, 2s]``. The idea is that in this way we can look at how channels with different signal strengths get processed differently according to the efficient coding prescription. And just for clarity: this is a completely arbitrary choice I have made for illustrative purposes only! We will see below where the values of ``s_k`` can come from when applying this efficient coding theory to a concrete case.

The other parameter we can manipulate in the interactive plot is the output budget ``Q``. We do so by setting the parameter ``\Lambda``. Remember that ``\Lambda`` is defined in such a way that ``\Lambda\rightarrow 0`` corresponds to a large output budget and ``\Lambda\rightarrow 1`` corresponds to a small output budget. Therefore, sliding the control to the right decreases the budget and sliding the control to the left increases the budget. You can see the function ``Q=Q(\Lambda)`` plotted in the top right panel of the display.

The plots has four panels:
- top left, the total output power ``Q`` as a function of the Lagrange multiplier ``\Lambda``, ``Q=Q(\Lambda)``.
- top right, the gain ``\sqrt{g_k}`` as a function of the input power of the channel ``s_k``
- bottom left, the output strength ``s_k\sqrt{g_k}`` as a function of ``s_k``
- bottom right, the input and output strength compared for each of the 15 channels in our system. These are compared to the "noise floor" (minimum amount of output power, regardless of the signal and the gain) ``n_c``, which we have set equal to 1.

#### What to pay attention to
By playing with the interactive plot, it should be possible to find **two different coding regimes**.
- **``\Lambda\rightarrow 1^-, s_k\gg 1``, Transmission limited regime:** When stimuli are strong enough and the main constraint is output noise/bandwidth limitation, weaker input signals should be amplified more than stronger ones: neural gain should decrease as signal variability (strength) increases.
- **``\Lambda\rightarrow 0^+, s_k\rightarrow 0``, Sampling limited regime:** When the main limitation is noise corrupting the input (weak signals and large budget), weaker signals cannot be recovered by amplification, so neural gain should increase as signal variability (strength) increases.
"""

# ╔═╡ 5edc3017-a0c4-433c-90f2-2d1d24c4bab7
begin
	function plot_gain_from_budget(Λ, input_strength)
		
		l = @layout [a b; c d]
		
	    
		n_channels = 15
	    input_std_min = input_strength*0.5
	    input_std_max = input_strength*2
	    s = LinRange(input_std_min,input_std_max,n_channels)    
	    
	    Λ_range = LinRange(0.05, 0.95, n_channels)
	
	    p1 = plot(Λ_range, [total_output_power(s, l) for l in Λ_range], color=:gray, legend=false)
	    scatter!([Λ], [total_output_power(s,Λ)], color=:red, legend=false)
		vline!([Λ], color=:red)
	    hline!([total_output_power(s,Λ)], color=:red)
	    xlabel!("Lagrange multiplier \$\\Lambda\$")
	    ylabel!("Total output variance \$Q\$")
	    title!("Output variance \"budget\" \$Q(\\Lambda)\$")

	    p2 = plot(s, gain.(s, Λ_range[1]), color=:gray, linewidth=0.2, legend=false)
	    for l in Λ_range[2:end]
	        plot!(s, gain.(s, l), color=:gray, linewidth=0.2, legend=false)
		end
	    plot!(s, gain.(s, Λ), color=:red)
	    title!("Neural gain")
	    xlabel!(L"Input strength for a given channel, $s_k$")
	    ylabel!(L"Channel gain, $g_k$")

		p3 = plot(s, s.*gain.(s, Λ_range[1]), color=:gray, linewidth=0.2, legend=false)
	    for l in Λ_range[2:end]
	        plot!(s, s.*gain.(s, l), color=:gray, linewidth=0.2, legend=false)
		end
	    plot!(s, s.*gain.(s, Λ), color=:red)
	    title!("Transfer function")
	    xlabel!("Input strength for a given channel, \$s_k\$")
	    ylabel!("Output strength \$s_k\\sqrt{g_k}\$")
	    
	    p4 = bar((0:length(s)-1).+0.2, s, bar_width=0.4, label="Input signal strength")
	    bar!((0:length(s)-1).-0.2, sqrt.(output_power.(s, Λ)), bar_width=0.4, label="Output dynamic range")
	    hline!([1], label="Noise floor", linewidth=2)
	    xticks!(([0, 7, 14], ["1", "8", "15"]))
	    xlabel!("Channel")
	    ylabel!("Dynamic range")

		plot(p1, p2, p3, p4, layout=l, size=(800,600))
		
	end
end

# ╔═╡ b1eef15f-03e5-4e22-8851-1ddb8fb0e239
md"""
Lagrange multiplier controlling the output variance budget: ``Λ=``$(@bind budget Slider(0.01:0.01:0.99, default=0.5, show_value=true))

log signal strength (global level): ``\log_{10}(s)=``$(@bind log_input_strength Slider(-2:0.01:2, default=0, show_value=true))
"""

# ╔═╡ 0376373b-4908-41cf-b7ce-0b377af11b89
md"""
The global signal strength is ``s=`` $(@sprintf("%.2f", 10^(log_input_strength))). Remember that the actual signal strength values ``s_k`` are 15 numbers distributed in regular intervals between ``s/2`` and ``2s``.
"""

# ╔═╡ bec54691-1db8-4149-8b3c-0f6768a8e6a4
plot_gain_from_budget(budget, 10^(log_input_strength))

# ╔═╡ 76c171fb-9253-4d37-90a6-b6d756d6530d
md"""
## A few pointers to classic work addressing the transmission-limited regime

The transmission-limited regime is the one that has historically received the most attention, as it links to/generalizes simpler notions of efficient coding as redundancy reduction (for instance that explored in [Laughlin's 1981 paper](https://doi.org/10.1515/znc-1981-9-1040), which we saw in class) which apply particularly well in the sensory periphery.

For instance, the phenomenon of **contrast gain control** can be conceptualized elegantly with an efficient coding approach. Sensory neurons are ofen observed to modulate the steepness of their response function, or "gain", as a function of the constrast (variability) of their input. In other words, in sensory contexts where the sensory features relevant for the neuron vary over a very broad range the neuron's transfer function becomes less steep, so that the (broad) input range can still all be mapped to the (fixed) dynamic range of the neuron. Conversely, when the input varies only over a very narrow range, the neuron's gain increases, again resulting in the (now, narrow) input range being mapped to the full dynamic range.

A couple of nice papers that address this topic experimentally in a way that can be mapped to the discussion above are [Chander and Chichilnisky 2001](https://doi.org/10.1523/JNEUROSCI.21-24-09904.2001) in retina and [Rabinowitz et al 2011](http://dx.doi.org/10.1016/j.neuron.2011.04.030) in auditory cortex.

In the following, however, we will focus on a series of studies that explored the relevance of the sampling limited regime in sensory cortex.
"""

# ╔═╡ 3da433c1-9d02-48e2-9da5-b1adb3d6f504
md"""
## Analysis of Caramellino et al 2021
### Introduction

We will now discuss the work of [Caramellino et al 2021](https://doi.org/10.7554/eLife.72081), which is part of a larger body of research including [Victor and Conte 2012](https://doi.org/10.1364/JOSAA.29.001313), [Hermundstad et al 2014](https://doi.org10.7554/eLife.03722), and [Tesileanu et al 2020](https://doi.org/10.7554/eLife.54347).

This paper studies the perception of *visual textures*. Intuitively, you can think of a visual texture as of the collection semi-structured visual patterns or motifs that characterize the surface of objects. Importantly, although textures could be composed by smaller discrete elements discernible at a close inspection, the texture is always made up by a large number of such elements. Some examples are "foliage", or "wood grain", or "grass", or "gravel" . Here are a couple of examples of visual textures, just to make the idea more concrete (from [Portilla and Simoncelli 2000](https://doi.org/10.1023/A:1026553619983)):

$(Resource("https://raw.githubusercontent.com/epiasini/bmitns/main/lecture_8/Portilla2000_examples.jpg", :width => 400))

For the purpose of this study, we will only consider visual features that correspond to correlations between neighboring pixels of an image. The idea that relatively simple correlation patterns are important for texture perception goes back to [Julesz 1962](https://doi.org/10.1109/TIT.1962.1057698). Moreover, we will only work with "binary" black and white images composed only of black or white pixels.

Texture processing is believed to happen mostly in intermediate visual cortical areas. In this situation, and in reference to the theoretical ideas discussed above, the "transmission budget" for the output is assumed to be high due to the dense connectivity within and across cortical circuits (compare to the retina, where the optic nerve forms a bottleneck for information transmission to cortex). At the same time, textures are here assumed to be represented by multi-point pixel correlations, and it may be hard to obtain an accurate estimate of these high-order statistics from an image patch. Therefore, we are in the **sampling limited** regime in the discussion above --- the regime where we have high budget but weak signal. In this regime, efficient coding predicts that visual cortex should be more more sensitive to correlation patterns that have larger variability across natural scenes, and less sensitive to less variable patterns. **We will test this prediction at the behavioral level, asking whether rats show higher visual sensitivity to artificial textures that contain patterns that are more variable in natural images, and vice versa.**

### Task description

Here's a figure from [Hermundstad et al, eLife 2014](https://doi.org/10.7554/eLife.03722) that explains how **pixel correlations (and, importantly, their variability from image patch to image patch)** are measured from a database of natural images.

$(Resource("https://raw.githubusercontent.com/epiasini/bmitns/main/lecture_8/Hermundstad2014_fig_1.png", :width => 800))

"""

# ╔═╡ f842c227-f4e4-4a8a-95b3-c9a161c5c5b7
begin

	const TextureSample = BitArray{2}

	function Base.show(io::IO, m::MIME"image/svg+xml", t::TextureSample)
		width = 8cm
		height = 8cm
		x_to_y_ratio = size(t,2)/size(t,1)
		if x_to_y_ratio > 1
			height /= x_to_y_ratio
		elseif x_to_y_ratio < 1
			width *= x_to_y_ratio
		end
		Δx = 1/size(t,2)
		Δy = 1/size(t,1)
		white = findall(t)
		black = findall( .~ t)
		x_white = Δx .* (( x -> x[2] ).(white) .- 1)
		y_white = Δy .* (( x -> x[1] ).(white) .- 1)
		x_black = Δx .* (( x -> x[2] ).(black) .- 1)
		y_black = Δy .* (( x -> x[1] ).(black) .- 1)
		composition = compose(
			context(),
			(context(),
			rectangle(x_white, y_white, [Δx], [Δy]),
			fill("white")),
			(context(),
			rectangle(x_black, y_black, [Δx], [Δy]),
			fill("black")))
		backend = SVG(io, width, height)
		draw(backend, composition)
	end


	"""
		sample_texture(statistic, level, height, width=height)

	Sample a maximum-entropy texture from the ensemble defined by the given statistic and the given level for that statistic.
	"""
	function sample_texture(statistic, level, height, width=height)
		
		if statistic == "γ"
			source = rand(height, width)
			sample::TextureSample = source .< (1+level)/2
		elseif statistic == "β—"
			sample = rand([0,1], height, width)
			for j ∈ 2:width
				parity = rand(height, 1) .< (1+level)/2
				new_column = @. ~ (sample[:,j-1] ⊻ parity)
				sample[:,j] = new_column
			end
		elseif statistic == "β|"
			sample = transpose(sample_texture("β—", level, width, height))
		elseif statistic == "β∖"
			sample = rand([0,1], height, width)
			for j ∈ 2:width
				parity = rand(height-1, 1) .< (1+level)/2
				sample[2:end,j] = @. ~ (sample[1:end-1,j-1] ⊻ parity)
			end
		elseif statistic == "β∕"
			sample = sample_texture("β∖", level, height, width)[:,end:-1:1]
		elseif statistic == "θ◿"
			sample = rand([0,1], height, width)
			for j ∈ 2:width
				for i ∈ 2:height
					parity = rand() < (1+level)/2
					sample[i,j] = sample[i,j-1] ⊻ sample[i-1,j] ⊻ parity
				end
			end
		elseif statistic == "θ◸"
			sample = sample_texture("θ◿", level, height, width)[end:-1:1,end:-1:1]
		elseif statistic == "θ◹"
			sample = sample_texture("θ◿", level, height, width)[end:-1:1,:]
		elseif statistic == "θ◺"
			sample = sample_texture("θ◿", level, height, width)[:,end:-1:1]
		elseif statistic == "α"
			sample = sample = rand([0,1], height, width)
			for j ∈ 2:width
				for i ∈ 2:height
					parity = rand() < (1+level)/2
					sample[i,j] = ~ sample[i,j-1] ⊻ sample[i-1,j-1] ⊻ sample[i-1,j] ⊻ parity
				end
			end
		end
		
		return sample
	end
end

# ╔═╡ e2063dee-b706-4147-aff8-a8575d7f97c2
md"""
In this study, the visual stimuli will be synthetic visual textures that have a certain degree of two-point, three-point, or four-point correlations, and that contain the minimum amount of statistical structure given that constraint. The idea is that we want to estimate rat sensitivity to specific pixel correlation patterns, so to reduce confounds to the minimum we will work with visual stimuli that in some sense "contain only" those correlation patterns. 

Such textures can be mathematically defined as **maximum-entropy textures** where we maximise the entropy of the texture seen as a random variable, subject to a constraint on the value of the desired correlation. This mathematical construction, due to [Victor and Conte, J Opt Soc Am 2012](https://doi.org/10.1364/JOSAA.29.001313), allows us to define a 10-dimensional **"texture space"** where each axis corresponds to a different correlation pattern. Each of these axes has a name, denoted by a greek letter and (optionally) a symbol. You can find the description of each of the coordinate axes in the figure reproduced above from Hermundstad 2014. Note that the origin of this texture space (that is, the point where each of the coordinates is 0) corresponds to the case of a so-called **"white noise"** texture, which is the trivial texture where each pixel is sampled independently as black or white with 50/50 probability.

The (hidden) function above implements Victor and Conte's algorithm for synthetic texture generation. To get a sense for how these textures look like, you can play with the generator below. 

Statistic to be sampled ("axis" in texture space): $(@bind statistic Select(["γ", "β—", "β|", "β∖", "β∕", "θ◿", "θ◸", "θ◹", "θ◺", "α"]))

Level of the statistic ("coordinate" in texture space): $(@bind level Slider(-1:0.01:1, default=0, show_value=true))
"""

# ╔═╡ 1b034394-1775-4ea8-8fc3-38832494c4d2
sample_texture(statistic, level, 100)

# ╔═╡ dfea496b-f89b-4845-8df5-fc11504499e3
md"""
The figure below is taken from [Caramellino et al, eLife 2021](https://doi.org/10.7554/eLife.72081) and explains the task in rats. This is a binary classification task (yes/no task). On each trial, the rat sees one image, which could be either a random sample of white noise (black and white pixels fully at random) or a sample from a maximum-entropy texture with a given intensity of a certain correlation (like the images you can generate above). The task of the rat is to report whether the image was fully random (white noise) or structured (texture). Conceptually, this allows to measure the sensitivity of the rat to different pixel statistics/texture patterns by seeing how far away, along different axes, the structured stimulus needs to be from the origin of the space (white noise) to be distinguishable from it.

$(Resource("https://raw.githubusercontent.com/epiasini/bmitns/main/lecture_8/Caramellino2021_fig_1.png", :width => 800))

"""

# ╔═╡ 0355c0dd-7151-4de1-b37b-dc981f0940fa
md"""
### Getting the data
Now that we have described the experiment, let's start by taking a look at the data. We're going to download the data published with the Caramellino paper, and to perform some preprocessing to rearrange it in a format that will be useful for us. The data is freely available [here](https://zenodo.org/doi/10.5281/zenodo.4762567), and the cell below will download it for us. Again here (as in other similar passages below) we are hiding the code by default but you can inspect it if you wish by downloading the notebook and opening it in Pluto.

The data is summarized in the table below, which identifies the rats (there are two "batches" of rats in the experiment, and within each batch the rats are identified by a rat ID), the name of the statistic that the rat was trained and tested on (gamma, beta, theta, alpha), the intensity ``s`` of the statistic (a number between 0 and 1, with 0 representing white noise and 1 representing the maximum value that that correlation can take), the number of trials ``T_s`` done on that rat at that intensity of that statistic, and the number of trials ``N_s`` where the rat reported "noise". 
"""

# ╔═╡ 7e4759f7-5507-4a15-9255-fd305ec9a1b3
begin
	filename = "rat_texture_task.csv"
	if !isfile(filename)
		Downloads.download(
			"https://zenodo.org/record/4763647/files/rat_texture_task.csv",
			filename)
	end
	data = DataFrame(CSV.File("rat_texture_task.csv"))
	data.success = data.success.==1

	possible_stimuli = [0 0.02 0.09 0.16 0.23 0.30 0.37 0.44 0.51 0.58 0.65 0.72 0.79 0.86 0.93 1]
	is_texture = data.statistic_intensity .!= 0
	data.report_noise = is_texture .⊻ data.success

	statistic_names = ["gamma", "beta", "theta", "alpha"]
	data.learned_statistic = statistic_names[data.learned_statistic]


	filter!(:statistic_intensity => ∈(possible_stimuli), data)
	filter!(:experiment_phase => ==(3), data)
	rename!(data, Dict("learned_statistic"=>"statistic"))

	data_gdf = groupby(data, ["rat_batch", "rat_ID", "statistic", "statistic_intensity"], sort=false)

	data = combine(data_gdf, :report_noise => (x -> [length(x) sum(x)]) => [:Ts, :Ns])
end

# ╔═╡ b32c0b97-4af8-402e-972d-0f6d5445ef7e
md"""
To interpret this data, we are going to build a Bayesian model of the perceptual process of the rat. We will simplify a bit the approach taken in Caramellino et al 2021, while keeping the essential components. Some of the text below is adapted from the paper.

### Step 1 - generative model
On any given trial, the nominal (true) value of the statistic is some value ``s``. The rats have to report whether the texture is white noise (``s=0``) or not. Note that in the experiment design only the positive axis of the texture space was used, so the two alternatives in practice are ``s=0`` and ``0<s<1``.

Because the texture has finite size, the empirical value of the statistic in the texture will be somewhat different from ``s``. We lump this uncertainty together with that induced by the animal’s perceptual process, and we say that any given trial results on the production of a percept ``x``, normally distributed around ``s``:
```math
p(x|s) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-s)^2}{2\sigma^2}\right]
```
Note that this means that when ``s=1`` the actual percept will be some ``x>1`` about half of the time. This doesn't really make sense given how the statistic is defined (because ``-1\leq s \leq 1`` by construction), but we will ignore this fact here for the sake of simplicity.

We will assume that each rat has some unknown prior over the alternatives (``s=0``, ``s>0``). We will parameterize the prior with the log odds:
```math
a := \ln\frac{p(s=0)}{p(s>0)}
```
More specifically, we assume that
  each rat assigns a prior probability ``p(s=0)=1/(1+e^{-a})`` to
  the presentation of a noise sample, and a probability of
  ``1/[K(1+e^{a})]`` to the presentation of a texture coming from
  any of the ``K`` nonzero statistic values. Note that this
  choice of prior matches the distribution actually used in generating
  the data for the experiment, except that ``a`` is a free
  parameter instead of being fixed at 0.
  
Therefore, **in our model we have two free parameters: ``a`` and ``\sigma``**, representing respectively the prior bias and the perceptual noise.

### Step 2 - inference
We define a decision variable $D$ as follows:
```math
  D(x) := \ln\frac{p(s=0|x)}{p(s>0|x)} =
  \ln\frac{p(x|s=0)}{p(x|s>0)} + \ln\frac{p(s=0)}{p(s>0)}
```
With this definition, the rat will report "noise" when $D>0$ and
"texture" otherwise.

By plugging in the likelihood functions and our choice of prior, we
get
```math
    D(x)= \ln\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{x^2}{2\sigma^2}\right] -\ln\left[\frac{1}{K}\sum_k\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-s_k)^2}{2\sigma^2}\right]\right] + a
```
which we can rewrite
```math
\tag{3}
    D(x)= a + \ln K -\frac{x^2}{2\sigma^2} -\ln\left[\sum_k\exp\left[-\frac{(x-s_k)^2}{2\sigma^2}\right]\right]
```

### Step 3 - response distribution
Now, remember that *given a value of the percept ``x``*, the
decision rule based on ``D`` is fully deterministic (maximum a
posteriori estimate). But on any given trial we don't know the value
of the percept --- we only know the nominal value of the statistic. On
the other hand, our assumptions above specify the distribution
``p(x|s)`` for any ``s``, so the deterministic mapping ``D(x)`` means that
we can compute the probability of reporting "noise" as
```math
    p(\text{report noise}|s) = p(D>0|s) = \int_{x:D(x)>0} p(x|s) \mathrm{d}x
```

Therefore, if we define ``x^*`` as the decision criterion in measurement space, i.e.,
```math
x^* = x^*(a,\sigma) \text{ such that } D(x^*)=0
```
the response distribution is, as usual, a cumulative Gaussian:
```math
p(\text{report noise}|s) = \int_{-\infty}^{x^*}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-s)^2}{2\sigma^2}\right]\mathrm{d}x = 1-\Phi\left[{\frac{s-x^{*}}{\sigma}}\right]
```

Importantly, note that unlike in the previous lectures of the course **we do not have a closed-form expression for ``x^*`` as a function of ``a`` and ``\sigma``**. However, given any values for ``a`` and ``\sigma`` we can plug them into Equation 3 and solve that equation for ``D(x)=0`` numerically. You can see this at play in the left panel of the plot below: the blue line is ``D(x)`` for the given values of ``a`` and ``\sigma``, and the horizontal red line is where ``D=0``. Finding the value of ``x^*`` numerically, essentially, means plotting the blue curve and seeing for what value of ``x`` it intersects the horizontal red line.

Note also that it is possible to find a combination of ``a`` and ``\sigma`` for which ``D(x)=0`` has no solution! Does this make sense?
"""

# ╔═╡ d9ae60c9-0fd8-4c7b-a2d3-4fd737b507fc
begin
	function D(x, p)
		(a, σ) = p
		possible_stimuli = 0.02:0.07:1.01
		K = length(possible_stimuli)
		a + log(K) - x^2/(2*σ^2) - logsumexp(-(x .- possible_stimuli).^2/(2*σ)^2)
	end

	function x_star(a, σ)
		try
			solution = find_zero(D, 0, [a σ])
		catch e
			# if σ is large enough and a is negative enough, the decision variable D is negative for all values of the measurement. In this regime, the measurement is so noisy and the prior is so shifted towards reporting the presence of the texture that the sensory evidence is completely disregarded, and the answer is always "texture" (never "noise"). In this case, the function call we use above to find x_star will fail with an error, because no zero of the D(x) function can be found. When this happens, we conventionally set x_star to negative infinity. When plugged into the expression for the response distribution, this will yield the desired effect that p(report noise|s)=0 for all values of s.
			if isa(e, Roots.ConvergenceFailed)
				if D(0, [a σ]) < 0
					solution = -Inf
				else
					solution = Inf
				end
			else
				throw(e)
			end
		end
	end

	function p_rep_noise(s, a, σ)
		return 1 - cdf(Normal(), (s-x_star(a, σ))/σ)
	end
end

# ╔═╡ a240c360-429f-4bd6-a3dd-be1843917ddb
begin
	function plot_response_distribution(a, σ)
		s_range = LinRange(0, 1, 50)
		p = plot(s_range, p_rep_noise.(s_range, a, σ), legend=false)
		xlabel!("True intensity of the statistic \$s\$")
		ylabel!("Probability of reporting 'noise' \$p(\\hat{s}=0|s)\$")
		ylims!(0,1)
		return p
	end

	function plot_decision_variable_and_response_distribution(a, σ)
		s_range = LinRange(-0.3, 1, 50)
		p1 = plot(s_range, D.(s_range, Ref([a, σ])), legend=false)
		hline!([0], color=:red)
		xs = x_star(a, σ)
		if isfinite(xs)
			vline!([xs], color=:red)
			annotate!(xs, ylims()[1], Plots.text("\$x^*\$", :red, :bottom, :left))
		end
		xlabel!("Measurement \$x\$")
		ylabel!("Decision variable \$D(x)\$")
		p2 = plot_response_distribution(a, σ)

		l = @layout [a{0.38w} b]
		plot(p1, p2, layout=l)
	end
end

# ╔═╡ dd829eb7-23a1-4b27-b9cf-82cfbf64226d
md"""
α = $(@bind a Slider(-1:0.01:1, default=0, show_value=true))

σ = $(@bind σ Slider(0.05:0.01:0.5, default=0.2, show_value=true))
"""

# ╔═╡ 83e037e2-582b-4a20-992f-dc9fec9106d9
plot_decision_variable_and_response_distribution(a, σ)

# ╔═╡ d5134a14-67e7-483d-961d-e9f0b7be30a9
md"""
### Step 4 - fitting the model
Independently for each rat, we infer a value of ``a`` and ``\sigma``
by maximising the likelihood of the data under the model above. More
in detail, for a given rat and a given statistic value ``s`` (including
0), we call ``N_s`` the number of times the rat reported "noise", and
``T_s`` the total number of trials. For a given fixed value of ``a``
and ``\sigma``, under the ideal observer model the likelihood of ``N_s``
will be given by a Binomial probability distribution for ``T_s`` trials
and probability of success given by the probability of reporting noise
computed above
```math
  p_s(N_s|a,\sigma) = \binom{T_s}{N_s}p(\text{rep
    noise}|s,a,\beta)^{N_s}\left(1-p(\text{rep noise}|s,a,\beta)\right)^{T_s-N_s}
```
Assuming that the data for the different values of ``s`` is
conditionally independent given ``a`` and ``\sigma``, the total log
likelihood for the data of the given rat is simply the sum of the log
likelihoods for the individual values of ``N_s``
```math
  LL := \ln p(\{N_{s_k}\}_{k=1}^K|a,\sigma) = \sum_{k=1}^K\ln p_{s_k}(N_k|a,\sigma)
```
Which we can rewrite
```math
\begin{align*}
LL &= \sum_{s}\left[\ln\binom{T_s}{N_s} + N_s\ln p(\text{rep
    noise}|s,a,\sigma) + (T_s-N_s) \ln\left(1-p(\text{rep
    noise}|s,a,\sigma)\right)\right]\\
    &= \sum_{s}\left[N_s\ln p(\text{rep
    noise}|s,a,\sigma) + (T_s-N_s) \ln\left(1-p(\text{rep
    noise}|s,a,\sigma)\right)\right] + C
\end{align*}
```
where ``C`` is a constant that does not depend on ``a`` and ``\sigma``.

#### Step 4.1 - defining the objective function to be minimized by the fit
Below we implement the calculation of ``LL`` given ``a``, ``\sigma`` and the data. Note that computing this function can be done numerically on a computer, but it wouldn't be possible to do it by hand in closed form because computing ``LL`` requires computing ``p(\text{rep noise}|s, a,\sigma)``, which in turn requires computing ``x^*`` (see definitions above).
"""

# ╔═╡ a4c03618-0ee9-419d-869d-3dd823c0ccf2
begin
	function LL(a, σ, s, Ts, Ns)
	    ps = p_rep_noise.(s, a, σ)
	    return sum(@. Ns * log(ps) + (Ts-Ns) * log(1-ps))
	end
end

# ╔═╡ 24920b7f-4022-4607-8460-f7d37dfb54b4
md"""
#### Step 4.2 - actually performing the fit

Now we actually fit the model to each rat. The result is summarized in the table below, which gives the estimated values of ``a`` and ``\sigma`` for each rat (note that in this data each rat was tested only on one statistic, so it's possible to keep track of which rats were "gamma rats", "beta rats", etc).
"""

# ╔═╡ 02999c65-c4bc-4923-bd46-956dfe6aa90b
begin
	# to fit the model independently to the data for each rat, we start by defining a function that takes one rat's data and returns the fitted parameters for that rat.
	function fit_model(s, Ts, Ns)
		# inside this function, we define a "loss function" to be minimized by the fitting procedure. This is just the (negative) log likelihood (defined in a cell above), with the data arguments s, Ts and Ns "frozen" to those of the rat under exam. The only actual arguments to the function are the model parameters a and σ. We pass these arguments to the function as a tuple (a, σ) because this is what the optimization procedure (the "optimize" function below) expects from the function to be optimized.
		function function_to_be_minimized(params)
		    a, σ = params
			return -LL(a, σ, s, Ts, Ns)
		end
		# for the particular optimization procedure we perform, we need to pick an initial guess for the solution. We pick a=0.1 and σ=0.4 as reasonable initial choices.
		initial_guess = [0.1, 0.4]
		# we can now find the minimum of our loss function. We do this by using the `optimize` function from the `Optim.jl` package, which by default uses the Nelder-Mead optimization algorithm (https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).
		solution = optimize(function_to_be_minimized, initial_guess)
		# the `optimize` function returns the solution as well as other information that we don't care too much about right now. We extract the values of a and σ that minimize the loss function and return those as output of our fit_model function.
		return [solution.minimizer[1] solution.minimizer[2]]
	end

	# the following expressions may look a bit cryptic, and there's no need for you to understand exactly why this is written this way (but if you want to do so a good place to start is to look into split-apply-combine operations on dataframes: https://dataframes.juliadata.org/stable/man/split_apply_combine/). Anyway, all they do is to take our data, split it in smaller tables each one of which contains only the data for one rat, and then use the data from each of those smaller tables to call the fit_model function we just defined. This results in a fitted value of a and σ for each rat, which we finally combine in another table (printed as the output of this code cell).
	grouped_data = groupby(data, (["rat_batch", "rat_ID", "statistic"]), sort=false)
	fits = combine(grouped_data, [:statistic_intensity, :Ts, :Ns] => fit_model => [:a, :σ])
end

# ╔═╡ 87cb2f98-14df-493c-bef9-154f622553ef
md"""
#### Visualize fit results

We can plot each rat's data in detail, together with its fitted model, to show that the model actually captures quite well the behavior of the animals.
"""

# ╔═╡ b9cc4989-4915-4c93-8d01-2172f6e5710a
function plot_rat_fit(rat_abs_number)
	function rat_filter(batch, ID, statistic)
		batch==fits[rat_abs_number,"rat_batch"] && ID==fits[rat_abs_number,"rat_ID"] &&
		statistic==fits[rat_abs_number,"statistic"]
	end
	rat_data = filter([:rat_batch, :rat_ID, :statistic] =>rat_filter, data)

	plot_response_distribution(fits[rat_abs_number,"a"], fits[rat_abs_number,"σ"])
	plot!(rat_data.statistic_intensity, rat_data.Ns./rat_data.Ts, seriestype=:scatter)
	annotate!((0.98, 0.95, Plots.text("Batch: $(fits[rat_abs_number,"rat_batch"]), rat: $(fits[rat_abs_number,"rat_ID"]), statistic: $(fits[rat_abs_number,"statistic"])\na=$(round(fits[rat_abs_number,"a"],digits=2)), σ=$(round(fits[rat_abs_number,"σ"],digits=2))", :right)))
	
end

# ╔═╡ f7e03313-4d88-4532-aafa-af8ab54c2fa6
md"""
Select global rat ID: $(@bind rat_abs_number Slider(1:nrow(fits), show_value=true))
"""

# ╔═╡ 868d4ec3-ef1e-4698-b46f-844c764943a6
plot_rat_fit(rat_abs_number)

# ╔═╡ d82a1862-fb10-4c39-94f6-950589299e5c
md"""
We can also look at the fits all at the same time (we don't plot the data here as it would be too messy).
"""

# ╔═╡ 71e38447-bff0-49a2-a03a-b800fca7049d
begin

	l = @layout [a b; c d]

	plots = []
	s_range = LinRange(0,1,50)

	for statistic in ["gamma", "beta", "theta", "alpha"]
		this_stat_rats = filter(:statistic => ==(statistic), fits)
		p = plot()
		for row in eachrow(this_stat_rats)
			rat_batch, rat_ID, statistic, a, σ = Vector(row)
			plot!(s_range, p_rep_noise.(s_range, a, σ), color=:black, legend=false)
		end
		xlabel!("True intensity of the statistic")
		ylabel!("Probability of reporting 'noise'")
		ylims!(0,1)
		annotate!((0.5, 0.95, Plots.text("Statistic: $(statistic)", :center, :top)))
		push!(plots, p)
	end

	plot(plots[1], plots[2], plots[3], plots[4], layout=l, size=(800,800))

end

# ╔═╡ c48d9b0c-eeb8-4f17-b3f5-731d812f22b6
md"""
A little reminder of what "gamma, beta, theta, alpha" mean:

$(Resource("https://raw.githubusercontent.com/epiasini/bmitns/main/lecture_8/Hermundstad2014_fig_1_extract.png", :width => 600))
"""

# ╔═╡ 320feafb-9821-4486-be69-a811b61376fe
md"""
Now we can summarize the plots above by plotting the **sensitivity** of each rat, defined as ``1/\sigma``:
"""

# ╔═╡ a36466c5-df6d-464e-bd2b-ae521c3e31ae
begin

	transform!(fits, :σ => (x -> 1 ./ x) => :sensitivity)

	plot([0], [1])
	for (k, statistic) in enumerate(["gamma", "beta", "theta", "alpha"])
		this_stat_rats = filter(:statistic => ==(statistic), fits)
		plot!(k .+ 0.3*(rand(nrow(this_stat_rats)).-0.5), this_stat_rats[:,"sensitivity"], seriestype=:scatter, legend=false)
	end
	xlabel!("Statistic")
	ylabel!("Sensitivity")
	xticks!(([1,2,3,4], ["gamma", "beta", "theta", "alpha"]))
end

# ╔═╡ 6fb30817-fb0e-4cb5-abe4-a4ba83e109ff
md"""
### Conclusion - comparing the fits with the predictions from the theory

To conclude, we can finally test our efficient coding hypothesis --- that rats should be more sensitive to patterns that are more variable in natural images. So we can plot on the same axis the sensitivities we just computed and the standard deviations of the distribution of the statistic intensities in natural images (after an appropriate normalization). While we are at it, we also include the data on **human** (not rat) sensitivity obtained with an analogous experiment by Hermundstad et al 2014.

$(Resource("https://raw.githubusercontent.com/epiasini/bmitns/main/lecture_8/Hermundstad2014_fig_3A.png", :width => 600))
"""

# ╔═╡ ca37fbfb-d715-4dcc-a1f3-7459d8904df7
begin
	# human sensitivity to correlated patterns from Hermundstad et al 2014.
	# Respectively beta (2-point), theta (3-point), and alpha (4-point)
	human = [0.88489849 0.29890596 0.35722528]
	
	# variablity of 2-point, 3-point and 4-point statistics in natural images
	images = [3.79 1.13 1.44]
	
	# compute average sensitivities for rats
	rat = combine(groupby(fits, :statistic), :sensitivity => mean).sensitivity_mean
	# discard sensitivity to gamma as that was not examined in humans
	rat = rat[2:end]
	
	# normalize all three sets of empirical data to put them all on the same scale
	human ./= norm(human)
	images ./= norm(images)
	rat ./= norm(rat)

	plot(images', markershape=:circle, label="image std. dev.")
	plot!(human', markershape=:square, label="human sensitivity")
	plot!(rat, markershape=:diamond, label="rat sensitivity")

	xticks!([1, 2, 3], ["\\beta (2-point)", "\\theta (3-point)", "\\alpha (4-point)"])

end

# ╔═╡ 42c70357-9178-46e5-bd0c-8a9bedba7b7f
md"""
We can see that the prediction of the theory seems to hold quite well. Moreover, the same prediction explains well this aspect of perception in both humans and rats. This is particularly pleasant, because it's what you'd expect for two species that share common machinery for processing mid-level visual features, and/or have evolved in similar visual environments.

(🎉🎉🎉🎉🎉)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"

[compat]
CSV = "~0.10.11"
Compose = "~0.9.5"
DataFrames = "~1.6.1"
Distributions = "~0.25.103"
LaTeXStrings = "~1.3.1"
LogExpFunctions = "~0.3.26"
Optim = "~1.7.8"
Plots = "~1.39.0"
PlutoUI = "~0.7.53"
Roots = "~2.0.22"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "3e0ba432a774033852041532988d3ee3f90d702e"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "16267cf279190ca7c1b30d020758ced95db89cd0"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.5.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "bf6570a34c850f99407b494757f5d7ad233a7257"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.5"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a6c00f894f24460379cb7136633cef54ac9f6f4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.103"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "35f0c0f345bff2c6d636f95fdb136323b5a796ef"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.7.0"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "f512dc13e64e96f703fd92ce617755ee6b5adf0f"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.8"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f6f85a2edb9c356b829934ad3caed2ad0ebbfc99"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.29"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8ec28846dbf846228a32de5a6912c63e2052e3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.53"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6842ce83a836fbbc0cfeca0b5a4de1a4dcbdb8d1"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.8"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "0f1d92463a020321983d04c110f476c274bafe2e"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.22"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "242982d62ff0d1671e9029b52743062739255c7e"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.18.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─fa084448-889d-11ee-3172-c3c4348be3b0
# ╟─3c9e3ee3-3398-4792-ba27-ea0636627faa
# ╠═7152e1e9-f83d-4600-8245-e09dca2efec4
# ╟─6141a7c6-02a5-442b-807d-de05e47e6e85
# ╟─5edc3017-a0c4-433c-90f2-2d1d24c4bab7
# ╟─b1eef15f-03e5-4e22-8851-1ddb8fb0e239
# ╟─0376373b-4908-41cf-b7ce-0b377af11b89
# ╟─bec54691-1db8-4149-8b3c-0f6768a8e6a4
# ╟─76c171fb-9253-4d37-90a6-b6d756d6530d
# ╟─3da433c1-9d02-48e2-9da5-b1adb3d6f504
# ╟─f842c227-f4e4-4a8a-95b3-c9a161c5c5b7
# ╟─e2063dee-b706-4147-aff8-a8575d7f97c2
# ╟─1b034394-1775-4ea8-8fc3-38832494c4d2
# ╟─dfea496b-f89b-4845-8df5-fc11504499e3
# ╟─0355c0dd-7151-4de1-b37b-dc981f0940fa
# ╟─7e4759f7-5507-4a15-9255-fd305ec9a1b3
# ╟─b32c0b97-4af8-402e-972d-0f6d5445ef7e
# ╠═d9ae60c9-0fd8-4c7b-a2d3-4fd737b507fc
# ╟─a240c360-429f-4bd6-a3dd-be1843917ddb
# ╟─dd829eb7-23a1-4b27-b9cf-82cfbf64226d
# ╟─83e037e2-582b-4a20-992f-dc9fec9106d9
# ╟─d5134a14-67e7-483d-961d-e9f0b7be30a9
# ╠═a4c03618-0ee9-419d-869d-3dd823c0ccf2
# ╟─24920b7f-4022-4607-8460-f7d37dfb54b4
# ╠═02999c65-c4bc-4923-bd46-956dfe6aa90b
# ╟─87cb2f98-14df-493c-bef9-154f622553ef
# ╟─b9cc4989-4915-4c93-8d01-2172f6e5710a
# ╟─f7e03313-4d88-4532-aafa-af8ab54c2fa6
# ╟─868d4ec3-ef1e-4698-b46f-844c764943a6
# ╟─d82a1862-fb10-4c39-94f6-950589299e5c
# ╟─71e38447-bff0-49a2-a03a-b800fca7049d
# ╟─c48d9b0c-eeb8-4f17-b3f5-731d812f22b6
# ╟─320feafb-9821-4486-be69-a811b61376fe
# ╟─a36466c5-df6d-464e-bd2b-ae521c3e31ae
# ╟─6fb30817-fb0e-4cb5-abe4-a4ba83e109ff
# ╠═ca37fbfb-d715-4dcc-a1f3-7459d8904df7
# ╟─42c70357-9178-46e5-bd0c-8a9bedba7b7f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
