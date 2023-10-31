#!/usr/bin/env python
# coding: utf-8

# In[6]:


from pydantic import BaseModel as bm


# In[7]:


class resume(bm):
    pdf_content: bytes


# In[ ]:




