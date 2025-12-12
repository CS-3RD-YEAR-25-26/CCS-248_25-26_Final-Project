"""
Facebook-Style Social Media App with Hate Speech Detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import torch
from datetime import datetime
from rnn_model import BiLSTMHateSpeechClassifier
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences


class SocialMediaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HateShield - AI-Protected Social Network")
        self.root.geometry("900x750")
        
        # Modern color palette
        self.colors = {
            'bg_primary': '#FAFBFC',      # Soft off-white background
            'bg_secondary': '#FFFFFF',     # Pure white for cards
            'accent_primary': '#5B7FFF',   # Modern blue accent
            'accent_hover': '#4A6DE5',     # Darker blue for hover
            'accent_light': '#E8EEFF',     # Light blue for subtle highlights
            'text_primary': '#1A1D29',     # Almost black for primary text
            'text_secondary': '#6B7280',   # Gray for secondary text
            'text_muted': '#9CA3AF',       # Light gray for muted text
            'border': '#E5E7EB',           # Light border color
            'success': '#10B981',          # Green for success
            'danger': '#EF4444',           # Red for danger/warnings
            'warning': '#F59E0B',          # Orange for warnings
            'shadow': '#D1D5DB',           # Subtle gray shadow
        }
        
        # Modern typography
        self.fonts = {
            'heading': ('Inter', 20, 'bold'),
            'subheading': ('Inter', 14, 'bold'),
            'body': ('Inter', 11),
            'small': ('Inter', 9),
            'button': ('Inter', 11, 'bold'),
        }
        
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Load model
        self.device = torch.device('cpu')
        self.vocab = Vocabulary.load('vocabulary.pkl')
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=False,
            remove_punctuation=False
        )
        
        # Initialize model
        self.model = BiLSTMHateSpeechClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            pad_idx=self.vocab.word2idx[self.vocab.PAD_TOKEN]
        ).to(self.device)
        
        checkpoint = torch.load('best_bilstm_model.pt', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Store posts
        self.posts = []
        
        # Create UI
        self.create_header()
        self.create_post_button()
        self.create_feed()
        
    def create_header(self):
        """Create header with app title"""
        header = tk.Frame(self.root, bg='#1877F2', height=60)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="📱 Social Media",
            font=('Segoe UI', 18, 'bold'),
            bg='#1877F2',
            fg='white'
        )
        title.pack(side='left', padx=20, pady=10)
        
        subtitle = tk.Label(
            header,
            text="Protected by PoTech Hate Speech Detection",
            font=('Segoe UI', 9),
            bg='#1877F2',
            fg='#E4E6EB'
        )
        subtitle.pack(side='left', padx=5, pady=10)
        
    def create_post_button(self):
        """Create 'What's on your mind?' button"""
        post_btn_frame = tk.Frame(self.root, bg='white', height=80)
        post_btn_frame.pack(fill='x', pady=10, padx=20)
        post_btn_frame.pack_propagate(False)
        
        # Profile icon
        profile_label = tk.Label(
            post_btn_frame,
            text="👤",
            font=('Segoe UI', 24),
            bg='white'
        )
        profile_label.pack(side='left', padx=15, pady=15)
        
        # Create post button
        post_btn = tk.Button(
            post_btn_frame,
            text="What's on your mind?",
            font=('Segoe UI', 11),
            bg='#F0F2F5',
            fg='#65676B',
            relief='flat',
            cursor='hand2',
            anchor='w',
            command=self.open_create_post_dialog
        )
        post_btn.pack(side='left', fill='both', expand=True, padx=(0, 15), pady=15)
        
    def create_feed(self):
        """Create scrollable feed area"""
        # Feed container
        feed_container = tk.Frame(self.root, bg='#F0F2F5')
        feed_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Canvas with scrollbar
        self.feed_canvas = tk.Canvas(feed_container, bg='#F0F2F5', highlightthickness=0)

        # The outer frame defines scrollregion height; inner column is centered
        self.feed_frame = tk.Frame(self.feed_canvas, bg='#F0F2F5')
        self.feed_frame.bind(
            '<Configure>',
            lambda e: self.feed_canvas.configure(scrollregion=self.feed_canvas.bbox('all'))
        )

        # Centered content column (fixed width) for cards - much wider now
        self.feed_column_width = 1200
        self.feed_column = tk.Frame(self.feed_frame, bg='#F0F2F5')
        self.feed_column.pack(pady=10, anchor='center')

        self._feed_window_id = self.feed_canvas.create_window((0, 0), window=self.feed_frame, anchor='n')

        def _center_feed_column(event=None):
            try:
                canvas_width = self.feed_canvas.winfo_width()
                x = max(0, (canvas_width - self.feed_column_width) // 2)
                self.feed_canvas.coords(self._feed_window_id, x, 0)
            except Exception:
                pass

        self.feed_canvas.bind('<Configure>', _center_feed_column)
        
        # Enable mousewheel scrolling without visible scrollbar
        def _on_mousewheel(event):
            self.feed_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.feed_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.feed_canvas.pack(side='left', fill='both', expand=True)
        
        # Welcome message
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Show welcome message when no posts"""
        # Clear existing cards/messages in the feed column
        for child in self.feed_column.winfo_children():
            child.destroy()

        # Welcome card (packs into centered column)
        welcome_frame = tk.Frame(self.feed_column, bg='white', relief='solid', borderwidth=1)
        welcome_frame.pack(fill='x', pady=10)
        
        welcome_text = tk.Label(
            welcome_frame,
            text="👋 Welcome! Share your thoughts with the community.\n\nOur AI will keep the community safe by filtering hate speech. \n\nhehe di gid ya makay-o ang layout kag center sir grr </3 \n\n pero functional ni siya sir! thank youuu gid sir enjoy!",
            font=('Segoe UI', 11),
            bg='white',
            fg='#65676B',
            justify='center',
            pady=30,
            padx=40,
            wraplength=1100
        )
        welcome_text.pack()
        
    def open_create_post_dialog(self):
        """Open modern create post dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Post")
        dialog.geometry("500x400")
        dialog.configure(bg=self.colors['bg_secondary'])
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Header with close button
        header_frame = tk.Frame(dialog, bg=self.colors['bg_secondary'])
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        title = tk.Label(
            header_frame,
            text="Create Post",
            font=self.fonts['heading'],
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        title.pack(side='left')
        
        # Thin separator
        separator = tk.Frame(dialog, height=1, bg=self.colors['border'])
        separator.pack(fill='x', padx=20)
        
        # User info
        user_frame = tk.Frame(dialog, bg=self.colors['bg_secondary'])
        user_frame.pack(fill='x', pady=16, padx=20)
        
        # Profile circle
        profile_bg = tk.Label(
            user_frame,
            text="👤",
            font=('Inter', 16),
            bg=self.colors['accent_light'],
            fg=self.colors['accent_primary'],
            width=2,
            height=1
        )
        profile_bg.pack(side='left', padx=(0, 12))
        
        user_info = tk.Frame(user_frame, bg=self.colors['bg_secondary'])
        user_info.pack(side='left')
        
        username = tk.Label(
            user_info,
            text="User",
            font=self.fonts['subheading'],
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        username.pack(anchor='w')
        
        privacy = tk.Label(
            user_info,
            text="🌐 Public",
            font=self.fonts['small'],
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        privacy.pack(anchor='w')
        
        # Text input area with border
        text_container = tk.Frame(dialog, bg=self.colors['bg_primary'])
        text_container.pack(fill='both', expand=True, padx=20, pady=(0, 12))
        
        text_input = scrolledtext.ScrolledText(
            text_container,
            font=self.fonts['body'],
            wrap='word',
            relief='flat',
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent_primary'],
            height=8,
            borderwidth=0,
            padx=12,
            pady=12
        )
        text_input.pack(fill='both', expand=True)
        text_input.focus()
        
        # Placeholder
        placeholder = "What's on your mind?"
        text_input.insert('1.0', placeholder)
        text_input.config(fg=self.colors['text_muted'])
        
        def on_focus_in(event):
            if text_input.get('1.0', 'end-1c') == placeholder:
                text_input.delete('1.0', 'end')
                text_input.config(fg=self.colors['text_primary'])
        
        def on_focus_out(event):
            if not text_input.get('1.0', 'end-1c').strip():
                text_input.insert('1.0', placeholder)
                text_input.config(fg=self.colors['text_muted'])
        
        def submit_post(event=None):
            """Submit post when Enter is pressed or Post button is clicked"""
            self.create_post(text_input.get('1.0', 'end-1c'), dialog, placeholder)
            return 'break'  # Prevent default Enter behavior
        
        text_input.bind('<FocusIn>', on_focus_in)
        text_input.bind('<FocusOut>', on_focus_out)
        # Bind Enter key to submit post (Shift+Enter for new line)
        text_input.bind('<Return>', submit_post)
        
        # Post button with hover effect
        post_btn = tk.Button(
            dialog,
            text="Post",
            font=self.fonts['button'],
            bg=self.colors['accent_primary'],
            fg='white',
            relief='flat',
            cursor='hand2',
            borderwidth=0,
            command=submit_post
        )
        post_btn.pack(fill='x', padx=20, pady=(0, 20), ipady=12)
        
        # Hover effects
        def btn_enter(e):
            post_btn.config(bg=self.colors['accent_hover'])
        
        def btn_leave(e):
            post_btn.config(bg=self.colors['accent_primary'])
        
        post_btn.bind('<Enter>', btn_enter)
        post_btn.bind('<Leave>', btn_leave)
        
    def create_post(self, text, dialog, placeholder):
        """Process and create post after hate speech check"""
        # Get text
        content = text.strip()
        
        # Check if empty or placeholder
        if not content or content == placeholder:
            messagebox.showwarning("Empty Post", "Please write something before posting.")
            return
        
        # Close the create post dialog
        dialog.destroy()
        
        # Show loading screen
        self.show_loading_screen(content)
    
    def detect_hate_speech(self, text):
        """Detect hate speech in text"""
        processed = self.preprocessor.preprocess(text)
        sequence = self.vocab.text_to_sequence(processed, max_length=100)
        
        if len(sequence) == 0:
            sequence = [self.vocab.word2idx[self.vocab.UNK_TOKEN]]
        
        padded_seq, length = pad_sequences([sequence], max_length=100)
        text_tensor = torch.LongTensor(padded_seq).to(self.device)
        length_tensor = torch.LongTensor(length).to(self.device)
        
        with torch.no_grad():
            output = self.model(text_tensor, length_tensor)
            probability = torch.sigmoid(output).item()
        
        # Lower threshold for better detection (0.5 is more balanced)
        threshold = 0.5
        is_hate = probability > threshold
        
        # Debug: print probability for testing
        print(f"Text: '{text}' | Probability: {probability:.4f} | Threshold: {threshold} | Is hate: {is_hate}")
        
        return is_hate, probability
    
    def show_loading_screen(self, content):
        """Show modern loading screen with progress bar"""
        loading = tk.Toplevel(self.root)
        loading.title("Analyzing...")
        loading.configure(bg='white')
        loading.resizable(False, False)
        loading.transient(self.root)
        loading.grab_set()
        
        # Content frame
        content_frame = tk.Frame(loading, bg='white', padx=50, pady=40)
        content_frame.pack()
        
        # Shield icon
        icon_label = tk.Label(
            content_frame,
            text="🛡️",
            font=('Segoe UI', 60),
            bg='white'
        )
        icon_label.pack(pady=(0, 20))
        
        # Title
        text_label = tk.Label(
            content_frame,
            text="Analyzing content...",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#1A1D29'
        )
        text_label.pack(pady=(0, 20))
        
        # Progress bar container with rounded corners
        progress_container = tk.Frame(content_frame, bg='white')
        progress_container.pack(pady=(0, 10))
        
        progress_bg = tk.Canvas(progress_container, bg='white', height=10, width=300, highlightthickness=0)
        progress_bg.pack()
        
        # Draw rounded background
        progress_bg.create_rectangle(0, 0, 300, 10, fill='#E5E7EB', outline='#E5E7EB', width=0)
        
        # Create rounded progress bar (will be updated)
        progress_bar = progress_bg.create_rectangle(0, 0, 0, 10, fill='#5B7FFF', outline='#5B7FFF', width=0)
        
        # Percentage label
        percent_label = tk.Label(
            content_frame,
            text="0%",
            font=('Segoe UI', 11),
            bg='white',
            fg='#6B7280'
        )
        percent_label.pack()
        
        # Center dialog relative to main window
        def center_dialog():
            loading.update_idletasks()
            self.root.update_idletasks()
            
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()
            dialog_width = loading.winfo_width()
            dialog_height = loading.winfo_height()
            
            x = root_x + (root_width - dialog_width) // 2
            y = root_y + (root_height - dialog_height) // 2
            
            loading.geometry(f"+{x}+{y}")
            loading.lift()
            loading.focus_force()
        
        # Position after a brief delay to ensure proper layout
        loading.after(10, center_dialog)
        
        # Animate progress bar with smooth easing
        self.loading_cancelled = False
        
        def animate_progress(progress=0):
            if loading.winfo_exists() and progress <= 100 and not self.loading_cancelled:
                # Smooth easing function (ease-out)
                eased_progress = 1 - (1 - progress / 100) ** 3
                width = int(300 * eased_progress)
                
                # Update progress bar with smooth animation
                progress_bg.coords(progress_bar, 0, 0, width, 10)
                percent_label.config(text=f"{int(progress)}%")
                # Faster animation - complete in ~1.5 seconds
                loading.after(15, lambda: animate_progress(progress + 1))
        
        # Start animation immediately
        loading.after(10, lambda: animate_progress(0))
        
        # Process after delay
        def process_content():
            try:
                is_hate, probability = self.detect_hate_speech(content)
                self.loading_cancelled = True
                
                if loading.winfo_exists():
                    loading.destroy()
                
                if is_hate:
                    self.show_violation_dialog(probability)
                else:
                    self.add_post_to_feed(content)
                    self.show_success_dialog(probability)
            except Exception as e:
                print(f"Error processing content: {e}")
                self.loading_cancelled = True
                if loading.winfo_exists():
                    loading.destroy()
        
        # Simulate processing time (1.5 seconds)
        loading.after(1500, process_content)
    
    def show_violation_dialog(self, probability):
        """Show modern hate speech violation dialog"""
        violation = tk.Toplevel(self.root)
        violation.title("Content Blocked")
        violation.configure(bg='white')
        violation.resizable(False, False)
        violation.transient(self.root)
        violation.grab_set()
        
        # Content frame
        content = tk.Frame(violation, bg='white', padx=50, pady=40)
        content.pack()
        
        # Warning icon
        icon_label = tk.Label(
            content,
            text="⚠️",
            font=('Segoe UI', 60),
            bg='white'
        )
        icon_label.pack(pady=(0, 20))
        
        # Title
        title = tk.Label(
            content,
            text="Content Blocked",
            font=('Segoe UI', 18, 'bold'),
            bg='white',
            fg='#EF4444'
        )
        title.pack(pady=(0, 15))
        
        # Message
        message = tk.Label(
            content,
            text=f"Your post violates our Community Standards on hate speech.\n\nWe don't allow content that attacks people based on their\nprotected characteristics or promotes violence.\n\nDetection Confidence: {probability*100:.1f}%",
            font=('Segoe UI', 11),
            bg='white',
            fg='#6B7280',
            justify='center'
        )
        message.pack(pady=(0, 25))
        
        # OK button
        ok_btn = tk.Button(
            content,
            text="I Understand",
            font=('Segoe UI', 12, 'bold'),
            bg='#EF4444',
            fg='white',
            relief='flat',
            cursor='hand2',
            borderwidth=0,
            command=violation.destroy,
            padx=40,
            pady=12
        )
        ok_btn.pack(ipady=8)
        
        def btn_enter(e):
            ok_btn.config(bg='#DC2626')
        
        def btn_leave(e):
            ok_btn.config(bg='#EF4444')
        
        ok_btn.bind('<Enter>', btn_enter)
        ok_btn.bind('<Leave>', btn_leave)
        
        # Center dialog relative to main window
        def center_dialog():
            violation.update_idletasks()
            self.root.update_idletasks()
            
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()
            dialog_width = violation.winfo_width()
            dialog_height = violation.winfo_height()
            
            x = root_x + (root_width - dialog_width) // 2
            y = root_y + (root_height - dialog_height) // 2
            
            violation.geometry(f"+{x}+{y}")
            violation.lift()
            violation.focus_force()
        
        # Position after a brief delay
        violation.after(50, center_dialog)
    
    def show_success_dialog(self, probability):
        """Show modern successful post dialog"""
        success = tk.Toplevel(self.root)
        success.title("Post Published")
        success.configure(bg='white')
        success.resizable(False, False)
        success.transient(self.root)
        success.grab_set()
        
        # Content frame
        content = tk.Frame(success, bg='white', padx=50, pady=40)
        content.pack()
        
        # Success icon
        icon_label = tk.Label(
            content,
            text="✅",
            font=('Segoe UI', 60),
            bg='white'
        )
        icon_label.pack(pady=(0, 20))
        
        # Title
        title = tk.Label(
            content,
            text="Post Published!",
            font=('Segoe UI', 18, 'bold'),
            bg='white',
            fg='#10B981'
        )
        title.pack(pady=(0, 15))
        
        # Message
        confidence = (1 - probability) * 100
        message = tk.Label(
            content,
            text=f"Your post is clean and safe for the community.\n\nSafety Score: {confidence:.1f}%",
            font=('Segoe UI', 12),
            bg='white',
            fg='#6B7280',
            justify='center'
        )
        message.pack(pady=(0, 25))
        
        # Close button
        close_btn = tk.Button(
            content,
            text="Close",
            font=('Segoe UI', 12, 'bold'),
            bg='#10B981',
            fg='white',
            relief='flat',
            cursor='hand2',
            borderwidth=0,
            command=success.destroy,
            padx=50,
            pady=12
        )
        close_btn.pack(ipady=8)
        
        def btn_enter(e):
            close_btn.config(bg='#059669')
        
        def btn_leave(e):
            close_btn.config(bg='#10B981')
        
        close_btn.bind('<Enter>', btn_enter)
        close_btn.bind('<Leave>', btn_leave)
        
        # Center dialog relative to main window
        def center_dialog():
            success.update_idletasks()
            self.root.update_idletasks()
            
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()
            dialog_width = success.winfo_width()
            dialog_height = success.winfo_height()
            
            x = root_x + (root_width - dialog_width) // 2
            y = root_y + (root_height - dialog_height) // 2
            
            success.geometry(f"+{x}+{y}")
            success.lift()
            success.focus_force()
        
        # Position after a brief delay
        success.after(50, center_dialog)
        
        # Auto close after 2 seconds
        success.after(2000, success.destroy)
    
    def add_post_to_feed(self, content):
        """Add post to feed"""
        # Clear welcome message if first post
        if len(self.posts) == 0:
            for widget in self.feed_column.winfo_children():
                widget.destroy()
        
        # Add to posts list
        timestamp = datetime.now()
        self.posts.insert(0, {'content': content, 'timestamp': timestamp})
        
        # Create post card inside the centered column
        post_frame = tk.Frame(self.feed_column, bg='white', relief='solid', borderwidth=1)
        post_frame.pack(fill='x', pady=8)
        
        # Post header
        header = tk.Frame(post_frame, bg='white')
        header.pack(fill='x', padx=15, pady=10)
        
        user_icon = tk.Label(
            header,
            text="👤",
            font=('Segoe UI', 20),
            bg='white'
        )
        user_icon.pack(side='left', padx=(0, 10))
        
        user_info = tk.Frame(header, bg='white')
        user_info.pack(side='left')
        
        username = tk.Label(
            user_info,
            text="User",
            font=('Segoe UI', 11, 'bold'),
            bg='white'
        )
        username.pack(anchor='w')
        
        # Time format
        time_str = self.format_timestamp(timestamp)
        time_label = tk.Label(
            user_info,
            text=f"{time_str} · 🌐",
            font=('Segoe UI', 9),
            bg='white',
            fg='#65676B'
        )
        time_label.pack(anchor='w')
        
        # Post content
        content_label = tk.Label(
            post_frame,
            text=content,
            font=('Segoe UI', 12),
            bg='white',
            fg='#050505',
            justify='left',
            wraplength=1150,
            anchor='w'
        )
        content_label.pack(fill='x', padx=15, pady=(0, 15))
        
        # Separator
        separator = tk.Frame(post_frame, height=1, bg='#CED0D4')
        separator.pack(fill='x', padx=15)
        
        # Actions (Like, Comment, Share)
        actions_frame = tk.Frame(post_frame, bg='white')
        actions_frame.pack(fill='x', pady=8)
        
        actions = [
            ('👍', 'Like'),
            ('💬', 'Comment'),
            ('↗️', 'Share')
        ]
        
        for icon, text in actions:
            action_btn = tk.Label(
                actions_frame,
                text=f"{icon} {text}",
                font=('Segoe UI', 10),
                bg='white',
                fg='#65676B',
                cursor='hand2'
            )
            action_btn.pack(side='left', expand=True)
    
    def format_timestamp(self, timestamp):
        """Format timestamp for display"""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            minutes = diff.seconds // 60
            return f"{minutes}m"
        elif diff.seconds < 86400:
            hours = diff.seconds // 3600
            return f"{hours}h"
        else:
            return timestamp.strftime("%B %d")


def main():
    root = tk.Tk()
    app = SocialMediaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
