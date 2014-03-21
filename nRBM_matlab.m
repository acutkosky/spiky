function e = nRBM_matlab()


%  x = [-400,-400,-400,400]
%  getrate(x)

  x = rand(884,10000)*400;
max(x);
t = rand(884,1)*400;
  hiddensize = 512
    weights = rand(hiddensize,884)*2-1;
s = cputime;
    rec = SampleRBM(t,weights,10);
q = cputime;
q-s
    weights = CD(x,weights,0.0001/hiddensize,0.0);
s = cputime;
s-q
rec = SampleRBM(t,weights,10);
q = cputime;
q-s

  
end

function s = sample(l)
  s = sign(l).*poissrnd(abs(l));
end

function spikes = GenSpikes(inputs,time)
  spikes = sample(getrate(inputs)*time);
end

function S = Synapse(spikes,weights)
  S = sample(weights*spikes);
end

function r = getrate(x)
  nA = 0.000000001;
  ms = 0.001;
  tau_ref = 2.5*ms;
  alpha = 17*nA;
  J_bias = 10*nA;
  tau_RC = 20*ms;
  J_th = 1*nA;

%  if alpha*x+J_bias < J_th
%    r = 0.0;
%  else
    r = 1.0./(tau_ref - tau_RC*log(1.0-J_th./(alpha*x+J_bias))) .* ((sign(alpha*x+J_bias - J_th)+1))/2;
%  end
end


function upweights = CD(data,weights,alpha,reg)
    time = 1.0/50.0;
    v0spikes = sample(data*time);
    h0spikes = GenSpikes(Synapse(v0spikes,weights)/time,time);
    v1spikes = GenSpikes(Synapse(h0spikes,weights')/time,time);

    h1spikes = GenSpikes(Synapse(v1spikes,weights)/time,time);

    update = alpha*(v0spikes*h0spikes' - v1spikes*h1spikes')';

    upweights = weights + update - weights*reg;
end

function rec = SampleRBM(data,weights,samples)
    time = 1.0/50.0;
				 av  = zeros(size(data));
				 size(av)		;		 
    for i = 1:samples
       v0spikes = sample(data*time);
       h0spikes = GenSpikes(Synapse(v0spikes,weights)/time,time);
       v1spikes = GenSpikes(Synapse(h0spikes,weights')/time,time);
size(v1spikes);
       av = av+v1spikes;
    end
    rec = av/(time*samples);
end
