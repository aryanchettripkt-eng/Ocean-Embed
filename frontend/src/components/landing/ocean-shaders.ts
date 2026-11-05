// Decorative ocean optics. These shaders never represent model data or observations.
const noise = `
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
  vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),
             mix(hash(i+vec2(0.,1.)),hash(i+1.),f.x),f.y);
}`;

export const plateVertex = `
varying vec2 vUv;
void main() {
  vUv=uv;
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
}`;

export const waterFragment = `
uniform sampler2D uLight, uDark;
uniform float uTime, uTheme;
varying vec2 vUv;
${noise}
void main() {
  vec2 uv=vUv;
  // The backdrop is stationary. Only illumination travels across the reef and water.
  vec3 base=mix(texture2D(uLight,uv).rgb,texture2D(uDark,uv).rgb,uTheme);
  vec3 sunlight=mix(vec3(1.,.79,.43),vec3(.24,.79,.86),uTheme);
  float shafts=0.;
  for(int i=0;i<7;i++) {
    float n=float(i);
    float origin=.45+n*.056+sin(uTime*.24+n*1.7)*.022;
    float angle=.17+n*.019+sin(uTime*.19+n)*.026;
    float width=.00012+(1.-uv.y)*(.0006+.00024*sin(uTime*.37+n));
    float axis=uv.x-origin+(1.-uv.y)*angle;
    float pulse=.55+.45*sin(uTime*.47+n*1.9);
    shafts+=exp(-axis*axis/width)*pulse;
  }
  float shaftMask=smoothstep(.04,.65,uv.y)*(1.-smoothstep(.87,1.,uv.y));
  base+=sunlight*shafts*shaftMask*.024;
  vec2 q=uv*vec2(72.,58.);
  q+=vec2(noise(q*.6+uTime*.19),noise(q*.7-uTime*.14))*1.6;
  float cells=sin(q.x+sin(q.y+uTime*.32))*cos(q.y-sin(q.x-uTime*.25));
  float caustic=pow(1.-abs(cells),13.);
  float floorMask=(1.-smoothstep(.08,.37,uv.y))*smoothstep(.04,.24,uv.x);
  float leftReef=(1.-smoothstep(.07,.21,uv.x))*(1.-smoothstep(.4,.65,uv.y));
  float rightReef=smoothstep(.78,.97,uv.x)*(1.-smoothstep(.18,.38,uv.y));
  float reef=clamp(floorMask+leftReef+rightReef,0.,1.);
  // Keep reflections broken and local; no large uniform brightness pulse.
  float reflectance=smoothstep(.01,.1,dot(base,vec3(.2126,.7152,.0722)));
  base+=sunlight*caustic*reef*reflectance*.014;
  gl_FragColor=vec4(base,1.);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export const surfaceVertex = `
uniform float uTime;
varying vec2 vImageUv;
varying vec3 vWaveNormal;
varying float vCrest;
void main() {
  // Project the resting ceiling onto the photograph so its edges blend with the plate.
  vec4 rest=projectionMatrix*modelViewMatrix*vec4(position,1.);
  vImageUv=rest.xy/rest.w*.5+.5;
  vec3 p=position;
  float a=p.x*9.+p.y*4.-uTime*1.05;
  float b=p.x*-6.+p.y*7.+uTime*.79;
  float c=p.x*19.+p.y*-11.-uTime*1.44;
  float fade=smoothstep(.64,.84,vImageUv.y);
  float height=sin(a)*.035+sin(b)*.022+sin(c)*.008;
  float dx=cos(a)*.315-cos(b)*.132+cos(c)*.152;
  float dy=cos(a)*.14+cos(b)*.154-cos(c)*.088;
  p.z+=height*fade;
  vWaveNormal=normalize(normalMatrix*vec3(-dx*fade,-dy*fade,1.));
  vCrest=height;
  gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
}`;

export const surfaceFragment = `
uniform sampler2D uLight, uDark;
uniform float uTime, uTheme;
varying vec2 vImageUv;
varying vec3 vWaveNormal;
varying float vCrest;
${noise}
void main() {
  vec3 normal=normalize(vWaveNormal);
  // Mirror beyond the photograph's edges instead of stretching its final pixel row.
  vec2 uv=1.-abs(mod(vImageUv+normal.xy*.002,2.)-1.);
  vec3 base=mix(texture2D(uLight,uv).rgb,texture2D(uDark,uv).rgb,uTheme);
  float alpha=smoothstep(.65,.79,vImageUv.y);
  float specular=pow(max(0.,dot(normal,normalize(vec3(.08,-.74,.67)))),24.);
  float shimmer=pow(noise(uv*vec2(155.,91.)+vec2(uTime*.65,-uTime*.31)),5.);
  float bright=smoothstep(.1,.65,max(base.r,max(base.g,base.b)));
  vec3 light=mix(vec3(1.,.8,.44),vec3(.35,.9,.96),uTheme);
  base*=.96+specular*.08;
  base+=light*bright*(specular*.045+shimmer*.2);
  gl_FragColor=vec4(base,alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export const fishVertex = `
uniform float uTime, uPhase, uSpeed;
varying vec2 vUv;
void main() {
  vUv=uv;
  vec3 p=position;
  float beat=uTime*uSpeed+uPhase;
  float tail=pow(uv.x,3.);
  p.y+=sin(beat+uv.x*5.)*tail*.105;
  p.z+=cos(beat+uv.x*4.)*tail*.12;
  p.x*=1.-pow(uv.x,2.)*(.025+.025*sin(beat));
  gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
}`;

export const fishFragment = `
uniform sampler2D uFish;
uniform float uOpacity, uTheme, uDepth;
varying vec2 vUv;
void main() {
  vec4 fish=texture2D(uFish,vUv);
  if(fish.a<.02) discard;
  float visibility=mix(.42,.94,uDepth)*mix(.92,.76,uTheme);
  vec3 haze=mix(vec3(.018,.12,.13),vec3(.002,.027,.035),uTheme);
  fish.rgb=mix(haze,fish.rgb*visibility,.5+uDepth*.5);
  gl_FragColor=vec4(fish.rgb,fish.a*uOpacity);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export const particleVertex = `
uniform float uTime, uPixelRatio;
attribute float aSeed;
varying float vOpacity;
void main() {
  vec3 p=position;
  p.x=mod(p.x+uTime*(.003+aSeed*.004)+1.,2.)-1.;
  p.y=mod(p.y+uTime*(.005+aSeed*.006)+1.,2.)-1.;
  vOpacity=.035+aSeed*.09;
  gl_PointSize=(.75+aSeed*1.2)*uPixelRatio;
  gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
}`;

export const particleFragment = `
uniform float uTheme;
varying float vOpacity;
void main() {
  float radius=length(gl_PointCoord-.5)*2.;
  if(radius>1.) discard;
  vec3 color=mix(vec3(1.,.86,.57),vec3(.45,.86,.89),uTheme);
  gl_FragColor=vec4(color,(1.-radius)*vOpacity);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;
