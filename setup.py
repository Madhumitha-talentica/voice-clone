from setuptools import setup, find_packages


setup(name='MyShell-OpenVoice',
      version='0.0.0',
      description='Instant voice cloning by MyShell.',
      long_description=open('README.md').read().strip(),
      long_description_content_type='text/markdown',
      keywords=[
            'text-to-speech',
            'tts',
            'voice-clone',
            'zero-shot-tts'
      ],
      url='https://github.com/myshell-ai/OpenVoice',
      project_urls={
        'Documentation': 'https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md',
        'Changes': 'https://github.com/myshell-ai/OpenVoice/releases',
        'Code': 'https://github.com/myshell-ai/OpenVoice',
        'Issue tracker': 'https://github.com/myshell-ai/OpenVoice/issues',
      },
      author='MyShell',
      author_email='ethan@myshell.ai',
      license='MIT License',
      packages=find_packages(),

      python_requires='>=3.9',
      install_requires=[
            # Updated numpy pin (>=1.22.4,<1.24) to satisfy pandas wheels used by gradio
            'numpy>=1.22.4,<1.24',
            'librosa==0.9.1',  # compatible with numpy <1.24
            'faster-whisper==0.9.0',
            'pydub==0.25.1',
            'wavmark==0.0.3',
            'eng_to_ipa==0.0.2',
            'inflect==7.0.0',
            'unidecode==1.3.7',
            'whisper-timestamped==1.14.2',
            'pypinyin==0.50.0',
            'cn2an==0.5.22',
            'jieba==0.42.1',
            'gradio==3.48.0',
            'langid==1.1.6',
            'pandas<2.2'  # ensure a pandas version compatible with numpy<1.24
      ],
      zip_safe=False
      )
