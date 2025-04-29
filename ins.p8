local x = { 1,2,3,foo="bar"}
x.x = x
x[" "]=cocreate(function() end)
setmetatable(x,x)

local f = function() end
x.f = f
x[f] = x
x[{x,f}]=f
x.l1={l2={l3={l4="!"}}}
-->8
do
local function qsort(t,f,l,r)
 if r-l<1 then return end
 local p=l
 for i=l+1,r do
  if f(t[i],t[p]) then
   if i==p+1 then
    t[p],t[p+1]=t[p+1],t[p]
   else
    t[p],t[p+1],t[i]=
     t[i],t[p],t[p+1]
   end
   p=p+1
  end
 end
 qsort(t,f,l,p-1)
 qsort(t,f,p+1,r)
end

local typew = {
 number=1,boolean=2,string=3,
 table=4, ['function']=5,
 userdata=6, thread=7
}

local function cmpkey(a,b)
 local ta,tb = type(a),type(b)
 if ta==tb and
  (ta=='string' or ta=='number')
  then return a < b
 end
 local wa,wb=typew[ta] or 32767,
             typew[tb] or 32767
 return wa==wb and
        ta<tb or wa<wb
end

local function getkeys(t)
 local slen=0
 while t[slen+1]~=nil do
  slen=slen+1
 end

 local keys, klen = {}, 0
 for k,_ in next,t do
  klen=klen+1
  keys[klen]=k
 end
 qsort(keys,cmpkey,1,klen)
 return keys,slen,klen
end

local function countref(x,ref)
 if type(x)~='table' then
  return
 end
 ref[x]=(ref[x] or 0)+1
 if ref[x]==1 then
  for k,v in next,x do
   countref(k,ref)
   countref(v,ref)
  end
  countref(getmetatable(x),ref)
 end
end

local function getid(x, ids)
 local id=ids[x]
 if not id then
  local t=type(x)
  id=(ids[t] or 0)+1
  ids[t],ids[x]=id,id
 end
 return id
end

local typesn={
 table="t",
 ["function"]="f", 
 thread="th",
 userdata="ud"
}

local function isident(x)
 if type(x)~= "string"
 or x=="" then
  return false
 end
 for i=1,#x do
  local c=ord(x,i)
  if(i==1 or c<48 or c>57) --0-9
  and (c<65 or c>90)  --lc a-z
  and (c<97 or c>122) --uc a-z
  and c~=95 then      -- _
   return false
  end
 end
 return true
end

local function tab(lvl)
 local s="\n"
 for _=1,lvl do s=s.." " end
 return s
end

local function x2s(x,d,lvl,
                   ids,ref)
 local tx=type(x)

 if tx=='string' then
  return '"'..x..'"'
 end

 if tx=='number' or tx=='nil'
 or tx=='boolean' then
  return tostr(x)
 end

 if tx=='table'
 and not ids[x] then
  if lvl >= d then
   return('{_}')
  end

  local s=""
  if ref[x] > 1 then
   s=s..'<'..getid(x,ids)..'>'
  end
  s=s..'{'

  local ks,slen,klen=getkeys(x)

  for i=1,klen do
   if i>1 then
    s=s..','
   end
   if i<=slen then
    s=s..
     x2s(x[i],d,lvl+1,ids,ref)
   else
    local k = ks[i]
    s=s..tab(lvl+1)..
     (isident(k) and k or
      "["..
      x2s(k,d,lvl+1,ids,ref)
      .."]"
     ).."=" ..
     x2s(x[k],d,lvl+1,ids,ref)
   end
  end

  local mt = getmetatable(x)
  if type(mt) == 'table' then
   if klen>0 then
    s=s..','
   end
   s=s..tab(lvl+1)..'<mt>=' ..
     x2s(mt,d,lvl+1,ids,ref)
  end

  if klen > slen
  or type(mt)=='table' then
   s=s..tab(lvl) -- last }
  end

  return s..'}'
 end

 -- function, userdata, thread,
 -- or previously visited table
 return '<' ..
  (typesn[tx] or tx) ..
  getid(x,ids) ..
  '>'
end

ins=function(root,depth)
  depth=depth or 32767

  local ref={}
  countref(root,ref)
  return x2s(root,depth,
             0,{},ref)
end
end
